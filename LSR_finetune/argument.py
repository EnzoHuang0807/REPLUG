import argparse
parser = argparse.ArgumentParser()


def add_training_args(parser):
      
    parser.add_argument('--epoch', type=int, default=3, 
                        help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=2e-5, 
                        help="Learning Rate")
    parser.add_argument('--lm_model_name_or_path', type=str, default="meta-llama/Llama-3.2-3B",
                        help="Path to directory containing LM model weights and config file")
    parser.add_argument('--dataset', type=str, default="wikitext-103-v1",
                        help="Dataset for LSR finetune")
    parser.add_argument('--dataset_ratio', type=float, default=1.0,
                        help="Ratio of dataset used for LSR finetune")
    parser.add_argument('--temperature_lm', type=float, default=0.1,
                        help="Temperature for LM likelihood")
    parser.add_argument('--temperature_re', type=float, default=0.1,
                        help="Temperature for retriever likelihood")
    parser.add_argument('--context_len', type=int, default=128,
                        help="Input context length per iteration")
    parser.add_argument('--max_seq_len', type=int, default=256,
                        help="Maximum context + prediction length")
    parser.add_argument('--batch_size', type=int, default=3,
                        help="Batch size for LSR finetune")
    parser.add_argument('--verbose', action="store_true",
                        help="If enabled, output training logs")

    return parser

def add_embeddings_args(parser):
    parser.add_argument('--per_gpu_batch_size', type=int, default=512, 
                        help="batch size for the passage encoder forward pass")
    parser.add_argument('--emb_dir', type=str, default='LSR_embeddings', 
                        help='directory path to save embeddings')
    parser.add_argument('--prefix', type=str, default='passages', 
                        help='prefix path to save embeddings')
    parser.add_argument('--shard_id', type=int, default=0, 
                        help="Id of the current shard")
    parser.add_argument('--num_shards', type=int, default=1, 
                        help="total number of shards")
    parser.add_argument('--passage_maxlength', type=int, default=512, 
                        help="maximum number of tokens in a passage")
    parser.add_argument('--model_name_or_path', type=str, default='./LSR_retriever', 
                        help="path to directory containing model weights and config file")
    parser.add_argument('--no_fp16', action='store_true', 
                        help="inference in fp32")
    parser.add_argument('--no_title', action='store_true', 
                        help="title not added to the passage body")
    parser.add_argument('--lowercase', action='store_true', 
                        help="lowercase text before encoding")
    return parser


def add_retriever_args(parser):
    # retrieval
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
    parser.add_argument('--question_maxlength', type=int, default=512,
                         help="Maximum number of tokens in a question")
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