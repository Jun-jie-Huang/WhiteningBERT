"""
This examples loads a pre-trained model and evaluates it on the STSbenchmark dataset

Usage:
python evaluation_stsbenchmark.py
OR
python evaluation_stsbenchmark.py model_name
"""
import sys
sys.path.append("../../")
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer,  SentencesDataset, LoggingHandler, evaluation, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SequentialEvaluator
from sentence_transformers.readers import STSBenchmarkDataReader

from sentence_transformers.models.MyPooling import EasyPooling, Layer2Pooling, LayerNPooling
from sentence_transformers.evaluation.WhiteningEmbeddingSimilarityEvaluator import WhiteningEmbeddingSimilarityEvaluator
from sentence_transformers.evaluation.SimilarityFunction import SimilarityFunction

import logging
import sys
import os
import torch
import numpy as np
import argparse
import itertools



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
parser.add_argument("--pooling", default="aver", type=str,
                    help="['cls', 'aver', 'max']")
parser.add_argument("--batch_size", default=256, type=int,)
parser.add_argument("--combination_num", default=1, type=int,)
parser.add_argument("--combi_start", default=1, type=int,)
parser.add_argument("--combi_end", default=769, type=int,)
parser.add_argument("--encoder_name", default='bert-base-uncased', type=str,
                    help="['bert-base-uncased', ''roberta-base]")
# encoder_name = '../training/nli/output/training_nli_bert-base-uncased-2021-01-10_14-44-13'
parser.add_argument("--sts_corpus", default="../datasets/stsbenchmark/", type=str,)
args = parser.parse_args()


target_eval_files = ['sts-b', 'sickr',
                     'sts12.MSRpar', 'sts12.MSRvid', 'sts12.SMTeuroparl', 'sts12.surprise.OnWN', 'sts12.surprise.SMTnews',
                     'sts13.FNWN', 'sts13.headlines', 'sts13.OnWN',
                     'sts14.deft-forum', 'sts14.deft-news', 'sts14.headlines', 'sts14.images', 'sts14.OnWN', 'sts14.tweet-news',
                     'sts15.answers-forums', 'sts15.answers-students', 'sts15.belief', 'sts15.headlines', 'sts15.images',
                     'sts16.answer-answer', 'sts16.headlines', 'sts16.plagiarism', 'sts16.postediting', 'sts16.question-question']
target_eval_tasks = ['sts-b', 'sickr', 'sts12', 'sts13', 'sts14', 'sts15', 'sts16']
target_eval_data_num = [1379, 4927,
                        750, 750, 459, 750, 399,
                        189, 750, 561,
                        450, 300, 750, 750, 750, 750,
                        375, 750, 375, 750, 750,
                        254, 249, 230, 244, 209, ]


if args.whitening:
    args.sts_corpus += "white/"
    target_eval_files = [f+"-white" for f in target_eval_files]

word_embedding_model = models.Transformer(args.encoder_name, model_args={'output_hidden_states': True, 'batch_size': args.batch_size})
if args.last2avg:
    pooling_model = Layer2Pooling(args.pooling, word_embedding_model.get_word_embedding_dimension(), layer_i=0, layer_j=12)
else:
    pooling_model = EasyPooling(args.pooling, word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

results = []
for embed_dim_i in range(args.combi_start, args.combi_end):
    logger.info("Pool:{}, Enc.:{}, White:{}, EmbedDim:{}".format(args.pooling, args.encoder_name, args.whitening, embed_dim_i))

    evaluators = {task: [] for task in target_eval_tasks}  # evaluators has a list of different evaluator classes we call periodically
    sts_reader = STSBenchmarkDataReader(os.path.join(script_folder_path, args.sts_corpus))
    for idx, target in enumerate(target_eval_files):
        output_filename_eval = os.path.join(script_folder_path, args.sts_corpus + target + "-test.csv")
        if args.whitening:
            evaluators[target[:5]].append(WhiteningEmbeddingSimilarityEvaluator.from_input_examples(sts_reader.get_examples(output_filename_eval), measure_data_num=target_eval_data_num[idx], name=target, main_similarity=SimilarityFunction.COSINE, embed_dim=embed_dim_i))
        else:
            evaluators[target[:5]].append(EmbeddingSimilarityEvaluator.from_input_examples(sts_reader.get_examples(output_filename_eval), name=target, main_similarity=SimilarityFunction.COSINE))

    _all_results = []
    logger_text = ""
    for task, sequential_evaluator in evaluators.items():
        result = model.evaluate(SequentialEvaluator(sequential_evaluator, main_score_function=lambda scores: np.mean(scores)))
        logger_text += "%.2f \t" % (result * 100)
        _all_results.append(result * 100)
    logger.info(" \t".join(target_eval_tasks) + " \tOverall.")
    logger.info(logger_text + "%.2f"%np.mean(_all_results))
    results.append((embed_dim_i, np.mean(_all_results)))

logger.info("***** Finishing evaluation *****")
logger.info("********** Evaluation Results Spearman Cor. **********")
for idx, (dim, result) in enumerate(results):
    line = "idx-{}\t Dim: {} \t".format(idx, dim)
    line += '%.3f'%result
    logger.info(line)


