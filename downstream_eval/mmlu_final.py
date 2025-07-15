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

from openai import OpenAI
from itertools import combinations

from retriever import Retriever
from argument import add_lm_args, add_retriever_args
from llm_scorer import pairwise_score, relevance_score
from utils import *

import random
random.seed(0)
torch.manual_seed(0)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY").strip()
if not OPENAI_API_KEY or not OPENAI_API_KEY.strip():
    raise ValueError("Missing OPENAI_API_KEY. Please set it as an environment variable.")
client = OpenAI(api_key=OPENAI_API_KEY)


def call_api(args, prompt, temp):

    # Tokenize
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Generate output
    with torch.no_grad():
        gen_output = model.generate(
            input_ids,
            do_sample=True,
            temperature=temp,
            max_new_tokens=args.maxlen,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True
        )

    # Decode generated text
    full_sequence = gen_output.sequences[0]
    gen_token_id = full_sequence[input_ids.shape[1]]
    decoded_output = tokenizer.decode(full_sequence[input_ids.shape[1]:], skip_special_tokens=True)
    
    stop_seq="\n"
    if stop_seq in decoded_output:
        decoded_output = decoded_output.split(stop_seq)[0]

    # Get log probabilities of the top-4 tokens
    with torch.no_grad():
        outputs = model(full_sequence.unsqueeze(0))
        logits = outputs.logits[:, input_ids.shape[1] - 1, :]
        log_probs = F.log_softmax(logits, dim=-1).squeeze(0)

    top_log_probs = {}
    top_values, top_indices = torch.topk(log_probs, 4)

    for val, idx in zip(top_values, top_indices):
        token_str = tokenizer.decode([idx.item()]).strip()
        top_log_probs[token_str] = val.item()

    # Calculate perplexity of the first generated token 
    first_logprob = log_probs[gen_token_id].item()
    perplexity = np.exp(first_logprob)

    return decoded_output, (top_log_probs, perplexity)


def inference_one_ex(args, counter, prompt_batch, score_batch, eg):
    all_outputs = []
    all_weighted_probs = []
    all_predictions = []
    
    for i, prompt in enumerate(prompt_batch):
        output, probs = call_api(args, prompt, temp=0.01)
        ans = output
        all_outputs.append(ans)
        all_weighted_probs.append(probs[1]*score_batch[i])
        if args.save_predictions:
            all_predictions.append({
                "emsemble_id": i,
                "ans": ans,
                "prompt": prompt,
                "top_log_probs": probs[0],
                "prob": probs[1],
                "re_score": score_batch[i]
            })
    
    ans2prob_list = defaultdict(list)
    for ans, prob in zip(all_outputs, all_weighted_probs):
        ans2prob_list[ans].append(prob)
    ans2prob = {k: sum(v) for k, v in ans2prob_list.items()}

    final_ans = max(ans2prob.items(), key=operator.itemgetter(1))[0]
    gold = eg["answer"]
    em = single_ans_em(final_ans, gold)

    prediction_log = {
        "example_id": counter,
        "predicted_ans": final_ans,
        "gold": gold,
        "em": em,
        "esb_predictions": all_predictions   
    } if args.save_predictions else None

    return em, prediction_log


def data_from_csv_to_list(dev_df):
    demos = []
    for i in range(len(dev_df)):
        # print(dev_df.iloc[i, 0])
        one_d = {}
        one_d["question"] = f"{dev_df.iloc[i, 0]}\n(A) {dev_df.iloc[i, 1]}\n(B) {dev_df.iloc[i, 2]}\n(C) {dev_df.iloc[i, 3]}\n(D) {dev_df.iloc[i, 4]}"
        one_d["answer"] = dev_df.iloc[i, 5]
        demos.append(one_d)
    return demos


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
    parser.add_argument("--data_dir", "-d", type=str, default="data")
    args = parser.parse_args()
    
    if args.save_predictions:
        assert(args.result_dir is not None)
    
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    if args.do_retrieval:
        retriever = Retriever(args)
    else:
        retriever = None

    global tokenizer, model, device
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # load dataset
    subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join(args.data_dir, "test")) if "_test.csv" in f])
    all_cors = {}
    all_counter = 0 
    all_em = 0

    
    for _, subject in tqdm(enumerate(subjects)):

        cors = []        
        subject_em = 0
        subject_predictions = [] if args.save_predictions else None

        '''
        data process
        '''
        print(f"subject: {subject}")
        train_df = pd.read_csv(os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None)[:args.shots]
        val_df = pd.read_csv(os.path.join(args.data_dir, "val", subject + "_val.csv"), header=None)
        test_df = pd.read_csv(os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None)

        # build demos
        demos = data_from_csv_to_list(train_df) 
        
        # build test examples
        if args.split == "test":
            test_set = data_from_csv_to_list(test_df)
        elif args.split == "val":
            test_set = data_from_csv_to_list(val_df)
        
        if args.truncate and len(test_set) > 10:
            test_set = test_set[:10]
        print("test_set: ", len(test_set))
        
        # evaluate
        counter = 0
        demo_questions = [d["question"].strip() for d in demos]
        pbar = tqdm(test_set)
        
        # build prompt
        prompt_demo = ""
        if args.prompt_method in ["closed-book", "open-book"]:
            for demo in demos:
                # concat the top-1 doc
                if args.prompt_method == "open-book":
                    docs, scores = retrieve_ex(demo, retriever)
                    prompt_demo += f"Knowledge: {docs[0]}\n"
                prompt_demo += "Question: " + demo["question"] + "\n"
                answer = demo["answer"]
                prompt_demo += "Answer: " + answer.strip() + "\n\n"

        # run over test examples
        for eg in pbar:
            
            prompt = prompt_demo
            pbar.set_description(f"Processing test examples from {subject}")
            if eg["question"].strip() in demo_questions:
                continue
            counter += 1
            all_counter += 1
            
            if len(eg["question"].split()) > 400:
                eg["question"] = ' '.join(eg["question"].split()[-400 : ])
            
            prompt_batch = []
            score_batch = []
            if args.prompt_method == "open-book":
                docs, scores = retrieve_ex(eg, retriever, args.llm_scorer)
                # contatenation
                for doc, score in zip(docs, scores):
                    prompt_cur = prompt
                    prompt_cur += f"Knowledge: {doc}" + "\n"
                    prompt_cur += "Question: " + eg["question"]  + "\n"
                    prompt_cur += "Answer:"
                    prompt_batch.append(prompt_cur)
                    score_batch.append(score)

            elif args.prompt_method == "closed-book":
                prompt += "Question: " + eg["question"]  + "\n"
                prompt += "Answer:"
                prompt_batch.append(prompt)
                score_batch.append(1)
            
            if args.llm_scorer != "pairwise":
                score_batch = softmax(score_batch).tolist()
            
            em, prediction_log = inference_one_ex(args, counter, prompt_batch, score_batch, eg)
            all_em += em
            subject_em += em 
            cors.append(em)

            if args.save_predictions:
                subject_predictions.append(prediction_log)

        '''
        evaluation
        '''
        
        acc = np.mean(cors)            
        all_cors[subject] = cors
        print ("\n\n")

        if args.save_predictions:
            print(f"{subject}[{args.split}] acc: {acc:.2f}")
            subject_results = {
                "subject": subject,
                "split": args.split,
                "acc": acc,
                "predictions": subject_predictions
            }
            out_json = os.path.join(args.result_dir, f"{subject}_results.json")
            with open(out_json, 'w') as o_f:
                json.dump(subject_results, o_f, indent=4)
                print(f"{subject} {args.split} predictions saved to {out_json}")
                print()

    print ("MMLU overall acc: {}/{}={}%".format(all_em, all_counter, (all_em / all_counter) * 100))

    if args.save_predictions:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(current_dir, "category.json"), "r") as f:
            cat_map = json.load(f)

        overall_results = {"overall_acc": all_em / all_counter}
        for category in cat_map:
            cat_cors = []
            for subject in cat_map[category]:
                if subject in all_cors:
                    cat_cors += all_cors[subject]
            overall_results[f"{category}_acc"] = np.mean(cat_cors)

        out_json = os.path.join(args.result_dir, "overall_results.json")
        with open(out_json, 'w') as o_f:
            json.dump(overall_results, o_f, indent=4)
    
    # if retriever is not None:
    #     retriever.dump_query2docs()


if __name__ == '__main__':
    main()