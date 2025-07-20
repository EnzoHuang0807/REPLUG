import argparse

parser = argparse.ArgumentParser()

def add_lm_args(parser):
    
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--maxlen', type=int, default=None, help='Maximum number of tokens to be generated')
    parser.add_argument('--shots', type=int, default=5, help='Number of demos to use in the prompt')
    
    parser.add_argument("--model_name_or_path", type=str, default=None, required=True,
                        help="Path to directory containing LM model weights and config file")
    parser.add_argument("--save_predictions", default=False, action="store_true",
                        help="If set, save detailed prediction on disk")
    parser.add_argument("--llm_scorer", type=str, default=None, choices=['pairwise', 'relevance'],
                        help="If set, use GPT-4o-mini to generate document score")
    parser.add_argument("--result_dir", type=str, default=None,
                        help="Directory to save detailed predictions")
    return parser


def add_retriever_args(parser):
    # retrieval
    parser.add_argument('--do_retrieval', action='store_true',
                        help="If enabled, retrieve documents with the model specified")
    parser.add_argument('--passages', type=str, required=True,
                        help='Path to passages (.tsv file)')
    parser.add_argument('--passages_embeddings', type=str, required=True,
                        help='Path to encoded passages')
    parser.add_argument('--n_docs', type=int, default=10,
                        help="Number of documents to retrieve per questions")
    parser.add_argument('--re_per_gpu_batch_size', type=int, default=64,
                        help="Retriever batch size")
    parser.add_argument('--normalize_text',
                        action='store_true', help="normalize text")
    parser.add_argument('--question_maxlength', type=int, default=512, help="Maximum number of tokens in a question")
    parser.add_argument('--cache_dict', type=str, default=None,
                        help='Path to cached query mappings (.pk file)') 
    parser.add_argument('--re_model_name_or_path', type=str, default="facebook/contriever",
                        help="Path to directory containing retriever model weights and config file")

    # index
    parser.add_argument('--projection_size', type=int, default=768)
    parser.add_argument("--n_subquantizers", type=int, default=0,
                        help='Number of subquantizer used for vector quantization, if 0 flat index is used')
    parser.add_argument("--n_bits", type=int, default=8,
                        help='Number of bits per subquantizer')
    parser.add_argument('--indexing_batch_size', type=int, default=1000000,
                        help="Batch size of the number of passages indexed")
    parser.add_argument("--save_or_load_index", action='store_true',
                        help='If enabled, save index and load index if it exists')
    parser.add_argument('--use_faiss_gpu', action="store_true", 
                        help='If enabled, use faiss GPU for retrieval inference')
    parser.add_argument('--num_gpus', type=int, default=-1)
    return parser