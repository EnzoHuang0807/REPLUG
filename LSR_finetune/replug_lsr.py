
import numpy as np
import os
from tqdm import tqdm
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from retriever import Retriever
from generate_passage_embeddings import generate_embeddings
from typing import Optional

import argparse
from datasets import load_dataset
import torch
import torch.nn as nn
from torch.optim import AdamW
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import utils

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model_config_path', required=True)
    parser.add_argument('--epoch', type=int, default=3, help="Number of training epoch")
    parser.add_argument('--data', default="wikitext-103-v1", type=str)
    parser.add_argument('--data_ratio', type=float, default=1.0)
    # parser.add_argument('--output_path', required=True)
    parser.add_argument('--max_docs', type=int, default=None)
    parser.add_argument('--doc_indices_path', type=str, default=None)
    parser.add_argument('--per_gpu_batch_size', type=int, default=64)

    parser.add_argument('--output_dir', type=str, default='finetune_embeddings', help='dir path to save embeddings')
    parser.add_argument('--prefix', type=str, default='passages', help='prefix path to save embeddings')
    parser.add_argument('--shard_id', type=int, default=0, help="Id of the current shard")
    parser.add_argument('--num_shards', type=int, default=1, help="Total number of shards")
    parser.add_argument('--passage_maxlength', type=int, default=512, help="Maximum number of tokens in a passage")
    parser.add_argument('--model_name_or_path', type=str, default="./finetune_retriever", help="path to directory containing model weights and config file")
    parser.add_argument('--no_fp16', action='store_true', help="inference in fp32")
    parser.add_argument('--no_title', action='store_true', help="title not added to the passage body")
    parser.add_argument('--lowercase', action='store_true', help="lowercase text before encoding")

    # retrieval
    parser.add_argument('--do_retrieval', type=int, default=1,
                        help="Number of documents to retrieve per questions")
    parser.add_argument('--use_faiss_gpu', action="store_true", 
                        help='If enabled, use faiss GPU for retrieval inference')
    parser.add_argument('--ensemble', type=int, default=0,
                        help="Number of documents to retrieve per questions")
    parser.add_argument('--passages', type=str, default="psgs_w100.tsv",
                        help='Path to passages (.tsv file)')  # wikitext-103-v1, wikitext-2-v1
    parser.add_argument('--passages_embeddings', type=str,
                        default="./finetune_embeddings/passages_00", help='Glob path to encoded passages')
    parser.add_argument('--n_docs', type=int, default=10,
                        help="Number of documents to retrieve per questions")
    parser.add_argument('--chunk_size', type=int, default=64,
                        help="Maximum number of words in a chunk")
    parser.add_argument('--normalize_text',
                        action='store_true', help="normalize text")
    parser.add_argument('--question_maxlength', type=int, default=128, help="Maximum number of tokens in a question")
    parser.add_argument('--random', type=int, default=0, help="random document")
    parser.add_argument('--cache_dict', type=str, default="./query2docs.pk",
                        help='Path to passages (.tsv file)') 
    parser.add_argument('--temperature_gold', type=float, default=0.1)
    parser.add_argument('--temperature_score', type=float, default=0.1)


    # 1024:
    parser.add_argument('--retrieved_max_length', type=int, default=256)
    parser.add_argument('--context_len', type=int, default=256)
    parser.add_argument('--pred_len', type=int, default=256)

    parser.add_argument('--re_model_name_or_path', type=str, default="facebook/contriever",
                        help="path to directory containing model weights and config file")

    parser.add_argument('--projection_size', type=int, default=768)
    parser.add_argument("--n_subquantizers", type=int, default=0,
                        help='Number of subquantizer used for vector quantization, if 0 flat index is used')
    parser.add_argument("--n_bits", type=int, default=8,
                        help='Number of bits per subquantizer')
    parser.add_argument('--indexing_batch_size', type=int, default=1000000,
                        help="Batch size of the number of passages indexed")
    parser.add_argument("--save_or_load_index", action='store_true',
                        help='If enabled, save index and load index if it exists')
    return parser.parse_args()

class LM:
    def get_perplexity_data(self, text) -> Optional[dict]:
        raise NotImplementedError

    @classmethod
    def create_from_config(cls, path):
        raise NotImplementedError

    def initialize_retriever(self, args):
        self.args = args
        if args.do_retrieval:
            self.retriever = Retriever(args)
        else:
            self.retriever = None

        
class GPT2LM(LM):

    def __init__(self, model_name, context_len=128, max_seq_len=256,
                 verbose=False, batch_size=1, optimizer=None, args=None, device=None):
        
        self.context_len = context_len
        self.max_seq_len = max_seq_len
        self.wb = utils.WaitBlocker()
        self.verbose = verbose
        self.tmp = 1
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.args = args
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Tokenizer & model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Set EOS token id (if available)
        if self.tokenizer.eos_token_id is not None:
            self.end_of_text_token_id = self.tokenizer.eos_token_id
        else:
            self.end_of_text_token_id = self.tokenizer.convert_tokens_to_ids(["<|endoftext|>"])[0]
        

    def forward_training(self, text):
        input_ids = self.tokenizer.encode_plus(text)["input_ids"]
        rolling_token_windows = utils.get_rolling_token_windows(
            token_list=input_ids,
            prefix_token=self.end_of_text_token_id,
            max_seq_len=self.max_seq_len,
            context_len=self.context_len,
        )

        batch_loss = []
        batch_index = 0
        # Remaining windows: input_tokens are context, pred_tokens are prediction
        for input_tokens, pred_tokens in rolling_token_windows:
            retriever_loss = self.forward_training_single(input_tokens, pred_tokens)
            batch_loss.append(retriever_loss)
            batch_index += 1
            if batch_index == self.batch_size:
                batch_loss = torch.stack(batch_loss)
                batch_loss = torch.mean(batch_loss)
                print(batch_loss.item())
                batch_loss.backward()
                batch_loss = []
                batch_index = 0
                self.optimizer.step()
                self.optimizer.zero_grad()
                

    def forward_training_single(self, input_tokens, pred_tokens):  
        query_id = input_tokens[:-len(pred_tokens)]
        # print("len(context):", len(query_id), "len(pred_tokens):", len(pred_tokens))
        query = self.tokenizer.decode(query_id)

        docs, scores = self.retriever.retrieve_passage([query])[0]
        plain_docs = [doc["text"] for doc in docs]

        # encode the retrieved docs
        questions_embedding = self.retriever.embed_queries_train([query]).to(self.device)
        passages_embedding = self.retriever.embed_queries_train(plain_docs).unsqueeze(0).to(self.device)
        retriever_score = torch.einsum("id, ijd->ij", [questions_embedding, passages_embedding])
        all_gold_score = []
        for i in range(len(docs)):
            doc_str = plain_docs[i]
            doc_encodings = self.tokenizer.encode(doc_str)
            input_tokens_tmp = doc_encodings + input_tokens
            block_output = self.get_token_logprobs(input_tokens=input_tokens_tmp, pred_tokens=pred_tokens)
            gold_score = block_output["logprobs"]
            all_gold_score.append(gold_score)
        all_gold_score = torch.FloatTensor(all_gold_score).to(self.device)
        retriever_loss = self.kldivloss(retriever_score, all_gold_score)
        return retriever_loss
          
    
    def kldivloss(self, score, gold_score):
        gold_score = torch.softmax(gold_score / self.args.temperature_gold, dim=-1)
        score = torch.nn.functional.log_softmax(score / self.args.temperature_score, dim=-1)
        return torch.nn.KLDivLoss()(score, gold_score)
    
    def get_perplexity_data(self, text) -> Optional[dict]:
        input_ids = self.tokenizer.encode_plus(text, return_tensors="pt")["input_ids"].squeeze(0).tolist()

        rolling_token_windows = utils.get_rolling_token_windows(
            token_list=input_ids,
            prefix_token=self.end_of_text_token_id,
            max_seq_len=self.max_seq_len,
            context_len=self.context_len,
        )

        all_logprobs = []
        all_positions = []

        for input_tokens, pred_tokens in tqdm(rolling_token_windows):
            query_id = input_tokens[:-len(pred_tokens)]
            print("len(context):", len(query_id), "len(pred_tokens):", len(pred_tokens))

            # Possibly do retrieval
            if self.args.do_retrieval and query_id:
                if self.args.random == 0:
                    query = self.tokenizer.decode(query_id)
                else:
                    query = "who is US president?"
                docs, scores = self.retriever.retrieve_passage([query])
                plain_docs = [doc["text"] for doc in docs]

                if self.args.ensemble == 0:
                    doc_str = "\n".join(plain_docs)
                    print(f"query: {[query]}\nretrieved doc: {[doc_str]}")
                    doc_encodings = self.tokenizer.encode(doc_str)[:self.args.retrieved_max_length]
                    input_tokens = doc_encodings + input_tokens
                    print("retrieve + context: ", len(input_tokens) - len(pred_tokens))
                    block_output = self.get_token_logprobs(input_tokens, pred_tokens)
                else:
                    logprobs_list = []
                    assert self.args.ensemble <= len(plain_docs)
                    for i in range(self.args.ensemble):
                        doc_str = plain_docs[i]
                        doc_encodings = self.tokenizer.encode(doc_str)[:self.args.retrieved_max_length]
                        input_tokens_tmp = doc_encodings + input_tokens
                        block_output = self.get_token_logprobs(input_tokens_tmp, pred_tokens)
                        logprobs_list.append(torch.FloatTensor(block_output["logprobs"]))
                    block_output["logprobs"] = (
                        torch.logsumexp(torch.stack(logprobs_list), dim=0) - np.log(len(logprobs_list))
                    ).numpy()
            else:
                block_output = self.get_token_logprobs(input_tokens, pred_tokens)

            all_logprobs.append(block_output["logprobs"])
            all_positions.append(block_output["positions"])

        if not all_logprobs:
            return None

        all_logprobs = np.concatenate(all_logprobs)
        all_positions = np.concatenate(all_positions)
        assert len(all_logprobs) == len(input_ids)
        return {
            "logprobs": all_logprobs,
            "positions": all_positions,
            "length": len(all_logprobs),
            "utf8_length": len(text.encode("utf-8")),
        }

    def get_token_logprobs(self, input_tokens, pred_tokens):
        device = self.device
        input_ids = torch.tensor(input_tokens + [pred_tokens[-1]], dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
                outputs = self.model(input_ids=input_ids)
                logits = outputs.logits  # (1, seq_len, vocab_size)

        # Compute logprobs of predicted tokens
        pred_start = input_ids.size(1) - len(pred_tokens)
        pred_token_ids = torch.tensor(pred_tokens, dtype=torch.long).to(device)

        logits = logits[0, pred_start - 1 : -1, :]  # align predictions with pred_token_ids
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        token_logprobs = log_probs[range(len(pred_token_ids)), pred_token_ids].cpu().numpy()

        positions = np.arange(pred_start - 1, pred_start - 1 + len(pred_token_ids))

        if self.verbose:
            print("Context:", self.tokenizer.convert_ids_to_tokens(input_tokens))
            print("Predicting:", self.tokenizer.convert_ids_to_tokens(pred_tokens))
            print("Perplexity:", np.exp(-token_logprobs.mean()))
            print()

        return {
            "logprobs": np.exp(-token_logprobs.mean()),
            "positions": positions,
        }


    @classmethod
    def create_from_config(cls, config):
        return cls(**config)


def main():
    args = parse_args()
    model = GPT2LM("meta-llama/Llama-3.2-3B", verbose=False, args=args)
    model.initialize_retriever(args)
    model.retriever.model.train()
    model.optimizer = AdamW(model.retriever.model.parameters(), lr=1e-3)

    if args.data.startswith("wikitext"):
        data = load_dataset("wikitext", args.data, split=f"test[0%:{int(args.data_ratio*100)}%]")
        data = data["text"]
        for i in range(args.epoch):
            for j in tqdm(range(len(data))):
                model.forward_training(data[j])

            model.retriever.model.save_pretrained("./finetune_retriever")
            model.retriever.tokenizer.save_pretrained("./finetune_retriever")
            generate_embeddings(args)
            model.initialize_retriever(args)

if __name__ == "__main__":
    main()




