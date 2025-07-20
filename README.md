# REPLUG: 
This includes a re-implementation of **REPLUG: Retrieval-Augmented Black-Box Language Models**

## Reasons for Re-Implementation

* The original implementation is currently unavailable and relied on several deprecated OpenAI API function calls.  
* In addition, it included only a partially completed implementation of the LSR (LM-Supervised Retrieval) finetuning.

## Highlights

- PyTorch-based re-implementation.
- Model loading and tokenization using HuggingFace's `transformers`.
- LSR finetuning support.
- Modular design for easy adaptation.

## LLM Scorer

* Instead of using cosine similarity between query and document embeddings as score, one can LLM's ability to rerank model based on relevance. 
* To enable this feature, use the `--llm_scorer` argument and set its value either to `pairwise` or `relevance`

### Methods

* `pairwise` : Generate every pairwise combinations of retrieved documents and prompt `gpt-4o-mini` to identify which document is more relevant to the question. Documents are ranked based on the results, while scores are assigned using the reciprocal of the rankings.
* `relevance` : Directly prompt `gpt-4o-mini` to assign a relevance score from 1 to 10 for each retrieved documents.

### Performance

Based on preliminary experiments, both LLM-based scoring methods outperforms the original cosine similarity baseline, with `pairwise` generally yeilding the best results. 

## Setup

```
conda env create -f environment.yml
conda activate replug
```

## Usage

### Preparation: Build Corpus Embeddings

The first step is to save embeddings of corpus. Download the Wikipedia file from:
```
wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
```

Generate embeddings for the corpus
```
EMB_DIR=/path/to/embeddings
python generate_passage_embeddings.py \
    --model_name_or_path facebook/contriever \
    --emb_dir $EMB_DIR  \
    --passages psgs_w100.tsv \
```

### Trivia QA

* Currently supports the data fields same as `mandarjoshi/trivia_qa`
* To use the LLM scorer, make sure the `OPENAI_API_KEY` is set.
* Full list of arguments can be found in `downstream_eval/arguments.py`

```
EMB_DIR=/path/to/embeddings
RES_DIR=/path/to/results

python -u downstream_eval/qa.py \
--split val \
--maxlen 10 \
--do_retrieval \
--re_model_name_or_path facebook/contriever \
--model_name_or_path meta-llama/Llama-3.2-3B \
--save_predictions \
--result_dir $RES_DIR \
--passages psgs_w100.tsv \
--save_or_load_index \
--passages_embeddings "$EMB_DIR/*" \
--llm_scorer pairwise
```


### MMLU

* Download MMLU data from https://github.com/hendrycks/test
* To use the LLM scorer, make sure the `OPENAI_API_KEY` is set.
* Full list of arguments can be found in `downstream_eval/arguments.py`

```
MMLU_DATA=/path/to/mmlu/data
EMB_DIR=/path/to/embeddings
RES_DIR=/path/to/results

python -u downstream_eval/mmlu.py \
--data_dir $MMLU_DATA \
--split test \
--maxlen 2 \
--do_retrieval \
--re_model_name_or_path facebook/contriever \
--model_name_or_path meta-llama/Llama-3.2-3B \
--save_predictions \
--result_dir $RES_DIR \
--passages psgs_w100.tsv \
--save_or_load_index \
--passages_embeddings "$EMB_DIR/*" \
--llm_scorer pairwise
```


### LSR Finetune

* Currently supports wikitext dataset subsets. See the [wikitext page](https://huggingface.co/datasets/EleutherAI/wikitext_document_level#data-instances) for available subsets.
* Full list of arguments can be found in `LSR_finetune/arguments.py`

```
EMB_DIR=/path/to/embeddings

python LSR_finetune/replug_lsr.py \
--passages psgs_w100.tsv \
--passages_embeddings "$EMB_DIR/*" \
--lm_model_name_or_path meta-llama/Llama-3.2-3B \
--re_model_name_or_path facebook/contriever \
--dataset wikitext-103-v1 \
--model_name_or_path LSR_retriever \
--emb_dir LSR_embeddings \
--epoch 3 \
--lr 2e-5 \
--temperature_re 0.1 \
--temperature_lm 0.1
```



