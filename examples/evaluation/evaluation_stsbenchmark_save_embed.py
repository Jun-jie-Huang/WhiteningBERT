"""
This examples loads a pre-trained model and evaluates it on the STSbenchmark, Sick-R and STS12-16 datasets.
Another function of this examples is to save the extracted word embeddings to `summary_path`

Usage:
python evaluation_stsbenchmark_save_embed.py --pooling aver --layer_index 1,12 --whitening --encoder_name bert-base-cased --summary_dir ./save_embeddings
"""
import sys
sys.path.append("../../")
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer,  SentencesDataset, LoggingHandler, evaluation, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SequentialEvaluator
from sentence_transformers.readers import STSBenchmarkDataReader

from sentence_transformers.evaluation.WhiteningEmbeddingSimilarityEvaluator import WhiteningEmbeddingSimilarityEvaluator
from sentence_transformers.models.MyPooling import EasyPooling, Layer2Pooling, LayerNPooling
from sentence_transformers.evaluation.SimilarityFunction import SimilarityFunction

import logging
import sys
import os
import torch
import numpy as np
import argparse

script_folder_path = os.path.dirname(os.path.realpath(__file__))
torch.set_num_threads(4)

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--whitening", action='store_true',
                    help="Whether to do whitening.")
parser.add_argument("--last2avg", action='store_true',
                    help="Whether to do avg-first-last.")
parser.add_argument("--wk", action='store_true',
                    help="Whether to do avg-first-last.")
parser.add_argument("--pooling", default="cls", type=str,
                    help="['cls', 'aver', 'max']")
parser.add_argument("--batch_size", default=256, type=int,)
parser.add_argument("--embed_dim", default=768, type=int,)
parser.add_argument("--encoder_name", default='bert-base-uncased', type=str,
                    help="['bert-base-uncased', ''roberta-base]")
parser.add_argument("--layer_index", default='12', type=str,
                    help="['bert-base-uncased', ''roberta-base]")
parser.add_argument("--sts_corpus", default="../datasets/stsbenchmark/", type=str,)
parser.add_argument("--summary_dir", default="./save_embeddings", type=str,)
args = parser.parse_args()

target_eval_files = ['sts-b', 'sickr',
                     'sts12.MSRpar',
                     'sts13.FNWN',
                     'sts14.deft-forum',
                     'sts15.answers-forums',
                     'sts16.answer-answer',]
target_eval_tasks = ['sts-b', 'sickr', 'sts12', 'sts13', 'sts14', 'sts15', 'sts16']

target_eval_data_num = [1379, 4927,
                        750,
                        189,
                        450,
                        375,
                        254,]

layer_index = [int(i) for i in args.layer_index.split(',')]
if args.whitening:
    args.sts_corpus += "white/"
    target_eval_files = [f+"-white" for f in target_eval_files]

word_embedding_model = models.Transformer(args.encoder_name, model_args={'output_hidden_states': True, 'batch_size': args.batch_size})

if args.last2avg:
    layer_index = [0, -1]
if args.wk:
    pooling_model = models.WKPooling(word_embedding_model.get_word_embedding_dimension())
    logger.info("wkpooling")
else:
    pooling_model = LayerNPooling(args.pooling, word_embedding_model.get_word_embedding_dimension(), layers=layer_index)

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

logger.info("Pool:{}, Encoder:{}, Whitening:{}".format(args.pooling, args.encoder_name, args.whitening))

evaluators = {task: [] for task in target_eval_tasks}         #evaluators has a list of different evaluator classes we call periodically
sts_reader = STSBenchmarkDataReader(os.path.join(script_folder_path, args.sts_corpus))
for idx, target in enumerate(target_eval_files):
    output_filename_eval = os.path.join(script_folder_path, args.sts_corpus + target + "-test.csv")
    if args.whitening:
        if len(evaluators[target[:5]]) == 0:
            summary_path = "{}/{}".format(args.summary_dir, args.encoder_name.replace("/", "="))
            if not os.path.exists(summary_path):
                os.makedirs(summary_path)
            summary_path = os.path.join(summary_path, target[:5]+".log")
            evaluators[target[:5]].append(WhiteningEmbeddingSimilarityEvaluator.from_input_examples(sts_reader.get_examples(output_filename_eval), measure_data_num=target_eval_data_num[idx], embed_dim=args.embed_dim, name=target, main_similarity=SimilarityFunction.COSINE, summary_path=summary_path))
        else:
            evaluators[target[:5]].append(WhiteningEmbeddingSimilarityEvaluator.from_input_examples(sts_reader.get_examples(output_filename_eval), measure_data_num=target_eval_data_num[idx], embed_dim=args.embed_dim, name=target, main_similarity=SimilarityFunction.COSINE))
    else:
        evaluators[target[:5]].append(EmbeddingSimilarityEvaluator.from_input_examples(sts_reader.get_examples(output_filename_eval), name=target, main_similarity=SimilarityFunction.COSINE))

all_results = []
logger_text = ""
for task, sequential_evaluator in evaluators.items():
    result = model.evaluate(SequentialEvaluator(sequential_evaluator, main_score_function=lambda scores: np.mean(scores)))
    logger_text += "%.2f \t"%(result*100)
    all_results.append(result*100)
logger.info(" \t".join(target_eval_tasks) + " \tOverall.")
logger.info(logger_text + "%.2f"%np.mean(all_results))




