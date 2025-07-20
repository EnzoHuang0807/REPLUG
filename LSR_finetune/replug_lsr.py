import os
import glob
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from retriever import Retriever
from generate_passage_embeddings import generate_embeddings
from argument import add_training_args, add_retriever_args, add_embeddings_args
from utils import *

        
class LM():

    def __init__(self, model_name, context_len=128, max_seq_len=256,
                 verbose=False, batch_size=3, optimizer=None, args=None, device=None):
        
        self.context_len = context_len
        self.max_seq_len = max_seq_len
        self.verbose = verbose
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
        
    def initialize_retriever(self):
        args = self.args
        self.retriever = Retriever(args)

    def forward_training(self, data, writer=None):

        batch_loss = []
        batch_index = 0
        t = 0

        for i in tqdm(range(len(data))):
            text = data[i]
            input_ids = self.tokenizer.encode_plus(text)["input_ids"]
            rolling_token_windows = get_rolling_token_windows(
                token_list=input_ids,
                prefix_token=self.end_of_text_token_id,
                max_seq_len=self.max_seq_len,
                context_len=self.context_len,
            )
        
            # Remaining windows: input_tokens are context, pred_tokens are prediction
            for input_tokens, pred_tokens in rolling_token_windows:
                retriever_loss = self.forward_training_single(input_tokens, pred_tokens)
                batch_loss.append(retriever_loss)
                batch_index += 1
                if batch_index == self.batch_size:
                    batch_loss = torch.stack(batch_loss)
                    batch_loss = torch.mean(batch_loss)

                    writer.add_scalar("Loss", batch_loss.item(), t)
                    writer.flush()
                    t += 1

                    batch_loss.backward()
                    batch_loss = []
                    batch_index = 0
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                

    def forward_training_single(self, input_tokens, pred_tokens):  
        query_id = input_tokens[:-len(pred_tokens)]
        query = self.tokenizer.decode(query_id)

        docs, _ = self.retriever.retrieve_passage([query])[0]
        plain_docs = [doc["text"] for doc in docs]

        # encode the retrieved docs
        questions_emb = self.retriever.embed_queries_train([query]).to(self.device)
        passages_emb = self.retriever.embed_queries_train(plain_docs).unsqueeze(0).to(self.device)
        
        # calculate cosine similarity
        questions_emb = torch.nn.functional.normalize(questions_emb, dim=-1)
        passages_emb = torch.nn.functional.normalize(passages_emb, dim=-1)
        re_score = torch.einsum("id, ijd->ij", [questions_emb, passages_emb])
        
        # LM probability
        lm_score = []
        for i in range(len(docs)):
            doc_str = plain_docs[i]
            doc_encodings = self.tokenizer.encode(doc_str)
            input_tokens_tmp = doc_encodings + input_tokens
            prob = self.get_token_prob(input_tokens=input_tokens_tmp, pred_tokens=pred_tokens)
            lm_score.append(prob)
        
        lm_score = torch.FloatTensor(lm_score).to(self.device)
        retriever_loss = self.kldivloss(re_score, lm_score)
        return retriever_loss
          
    
    def kldivloss(self, re_score, lm_score):
        re_score = torch.nn.functional.log_softmax(re_score / self.args.temperature_re, dim=-1)
        lm_score = torch.softmax(lm_score / self.args.temperature_lm, dim=-1)
        return torch.nn.KLDivLoss(reduction='batchmean')(re_score, lm_score)
    

    def get_token_prob(self, input_tokens, pred_tokens):
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

        if self.verbose:
            print("Context:", self.tokenizer.convert_ids_to_tokens(input_tokens))
            print("Predicting:", self.tokenizer.convert_ids_to_tokens(pred_tokens))
            print("Perplexity:", np.exp(-token_logprobs.mean()))
            print()

        return np.exp(token_logprobs.mean())


    @classmethod
    def create_from_config(cls, config):
        return cls(**config)


def main():
    
    parser = argparse.ArgumentParser()
    parser = add_retriever_args(parser)
    parser = add_embeddings_args(parser)
    parser = add_training_args(parser)
    args = parser.parse_args()

    model = LM(args.lm_model_name_or_path, args.context_len, args.max_seq_len,
               args.verbose, args.batch_size, args=args)
    model.initialize_retriever()
    model.retriever.model.train()
    model.optimizer = AdamW(model.retriever.model.parameters(), lr=args.lr)

    
    if args.dataset.startswith("wikitext"):
        data = load_dataset("wikitext", args.dataset, split=f"test[0%:{int(args.dataset_ratio*100)}%]")
        data = data["text"]

        for i in range(args.epoch):
            writer = SummaryWriter(f"LSR/epoch_{i+1}")
            model.forward_training(data, writer)

            model.retriever.model.save_pretrained(args.model_name_or_path)
            model.retriever.tokenizer.save_pretrained(args.model_name_or_path)

            args.re_model_name_or_path = args.model_name_or_path
            generate_embeddings(args)
            args.passages_embeddings = f"{args.embed_dir}/*"
            model.initialize_retriever()

if __name__ == "__main__":
    main()




