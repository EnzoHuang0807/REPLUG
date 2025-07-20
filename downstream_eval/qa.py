import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import pandas as pd
import json
import random
import torch
import torch.nn.functional as F
import operator

from tqdm import tqdm
from scipy.special import softmax
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from retriever import Retriever
from argument import add_lm_args, add_retriever_args
from llm_scorer import pairwise_score, relevance_score
from utils import *

import random
random.seed(0)
torch.manual_seed(0)


def call_api(args, prompt, temp):

    # Tokenize
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Generate output with log probabilities
    with torch.no_grad():
        gen_output = model.generate(
            input_ids,
            max_new_tokens=args.maxlen,
            pad_token_id=tokenizer.eos_token_id,
            temperature=temp,
            do_sample=True,
            return_dict_in_generate=True,
            output_scores=True
        )

    # Get generated token IDs (excluding prompt)
    full_sequence = gen_output.sequences[0]
    generated_ids = full_sequence[input_ids.shape[1]:]
    decoded_output = tokenizer.decode(generated_ids, skip_special_tokens=True)
    decoded_output = decoded_output.strip()

    # Extract log probability
    top_log_probs = []
    with torch.no_grad():
        outputs = model(full_sequence.unsqueeze(0))
        logits = outputs.logits[:, input_ids.shape[1] -1 : -1, :]
        log_probs = F.log_softmax(logits, dim=-1).squeeze(0)
    
    for i in range(log_probs.shape[0]):
        token_id = generated_ids[i].item()
        log_prob = log_probs[i, token_id].item()
        token_str = tokenizer.decode([token_id])
        if token_str == tokenizer.eos_token:
            break
        top_log_probs.append(log_prob)

    probability = np.exp(np.mean(top_log_probs))
    return decoded_output, (top_log_probs, probability)


def inference_one_ex(args, counter, prompt_batch, score_batch, eg):
    all_outputs = []
    all_probs = []
    all_predictions = []

    for i, prompt in enumerate(prompt_batch):
        output, probs = call_api(args, prompt, temp=0.01)
        ans = output
        all_outputs.append(ans)
        all_probs.append(probs[1]*score_batch[i])
        if args.save_predictions:
            all_predictions.append({
                "emsemble_id": i,
                "ans": ans,
                "prompt": prompt,
                "prob": float(probs[1]),
                "re_score": float(score_batch[i])
            })
    
    ans2prob_list = defaultdict(list)
    for ans, prob in zip(all_outputs, all_probs):
        ans2prob_list[ans].append(prob)
    ans2prob = {k: sum(v) for k, v in ans2prob_list.items()}
    
    final_ans = max(ans2prob.items(), key=operator.itemgetter(1))[0]
    gold = eg["answer"]["aliases"]

    em = single_ans_em(final_ans, gold)
    prediction_log = {
        "example_id": counter,
        "predicted_ans": final_ans,
        "gold": gold,
        "em": em,
        "esb_predictions": all_predictions   
    } if args.save_predictions else None
    return em, prediction_log


        
def retrieve_ex(demo, retriever, llm_scorer=None):
    query = demo["question"]
    docs, scores = retriever.retrieve_passage([query])[0]
    plain_docs = [doc["text"] for doc in docs]

    if llm_scorer != None:
        if llm_scorer == "pairwise":
            new_scores = pairwise_score(query, plain_docs)
        elif llm_scorer == "relevance":
            new_scores = relevance_score(query, plain_docs)
        scores = new_scores if new_scores != [] else scores
    return plain_docs, scores


def main():

    parser = argparse.ArgumentParser()
    parser = add_lm_args(parser)
    parser = add_retriever_args(parser)
    parser.add_argument("--truncate", type=int, default=500,
                         help="Truncate data to the specified size ; use -1 to disable truncation.")
    parser.add_argument("--dataset", type=str, default="mandarjoshi/trivia_qa",
                         help="Name of the dataset to evaluate.")
    parser.add_argument("--subset", type=str, default="rc",
                         help="Subset of the dataset.")
    
    args = parser.parse_args()

    if args.do_retrieval:
        retriever = Retriever(args)
    else:
        retriever = None

    args = parser.parse_args()
    if args.save_predictions:
        assert(args.result_dir is not None)

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    global tokenizer, model, device
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    predictions = []
    all_counter = 0 
    all_em = 0

    '''
    data process
    '''     
    demos = load_dataset(args.dataset, args.subset, split="train").select(range(args.shots))
    if args.split == "test":
        test_set = load_dataset(args.dataset, args.subset, split="test")
    elif args.split == "val":
        test_set = load_dataset(args.dataset, args.subset, split="validation")
    
    if args.truncate != -1 and len(test_set) > args.truncate:
            test_set = test_set.select(range(args.truncate))
    print("test_set: ", len(test_set))
    

    # evaluate
    demo_questions = [d["question"].strip() for d in demos]
    pbar = tqdm(test_set)

    # build prompt
    prompt_demo = ""
    for demo in demos:
        # concat the top-1 doc
        if args.do_retrieval:
            assert(retriever != None)
            docs, scores = retrieve_ex(demo, retriever)
            prompt_demo += f"Knowledge: {docs[0]}\n"
        prompt_demo += "Question: " + demo["question"] + "\n"
        answer = demo["answer"]["aliases"][0]
        prompt_demo += "Answer: " + answer.strip() + "\n\n"

    # run over test examples
    for eg in pbar:
       
        prompt = prompt_demo
        if eg["question"].strip() in demo_questions:
            continue
        all_counter += 1
        
        if len(eg["question"].split()) > 400:
            eg["question"] = ' '.join(eg["question"].split()[-400 : ])
        
        prompt_batch = []
        score_batch = []
        if args.do_retrieval:
            assert(retriever != None)
            docs, scores = retrieve_ex(eg, retriever)
            # contatenation version
            for doc, score in zip(docs, scores):
                prompt_cur = prompt
                prompt_cur += f"Knowledge: {doc}" + "\n"
                prompt_cur += "Question: " + eg["question"]  + "\n"
                prompt_cur += "Answer:"
                prompt_batch.append(prompt_cur)
                score_batch.append(score)

        else:
            prompt += "Question: " + eg["question"]  + "\n"
            prompt += "Answer:"
            prompt_batch.append(prompt)
            score_batch.append(1)

        if args.llm_scorer != "pairwise":
                score_batch = softmax(score_batch).tolist()
       
        em, prediction_log = inference_one_ex(args, all_counter, prompt_batch, score_batch, eg)
        all_em += em
        if args.save_predictions:
                predictions.append(prediction_log)

    print ("QA overall acc: {}/{}={}%".format(all_em, all_counter, (all_em / all_counter) * 100))

    if args.save_predictions:
        results = {
            "acc": float(all_em / all_counter),
            "predictions": predictions
        }

        out_json = os.path.join(args.result_dir, f"overall_results.json")
        with open(out_json, 'w') as o_f:
            json.dump(results, o_f, indent=4)
            print(f"predictions saved to {out_json}")
            print()

    if retriever is not None:
        retriever.dump_query2docs()


if __name__ == '__main__':
    main()