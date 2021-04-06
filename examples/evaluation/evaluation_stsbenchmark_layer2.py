"""
This examples loads a pre-trained model and evaluates it on the STSbenchmark, Sick-R and STS12-16 datasets.
This examples will enumerate all pssible combinations of two layers and automatically evaluate the combinations consequently.

Usage:
python evaluation_stsbenchmark_layer2.py --pooling aver --whitening --encoder_name bert-base-cased
Note to modify the total number of layers, i.e. `total_layers` at line 70, according to the PLM.
"""
import sys
sys.path.append("../../")
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer,  SentencesDataset, LoggingHandler, evaluation, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SequentialEvaluator
from sentence_transformers.readers import STSBenchmarkDataReader

from sentence_transformers.models.MyPooling import EasyPooling, Layer2Pooling
from sentence_transformers.evaluation.WhiteningEmbeddingSimilarityEvaluator import WhiteningEmbeddingSimilarityEvaluator
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
parser.add_argument("--pooling", default="cls", type=str,
                    help="['cls', 'aver', 'max']")
parser.add_argument("--batch_size", default=256, type=int,)
parser.add_argument("--encoder_name", default='bert-base-uncased', type=str,
                    help="['bert-base-uncased', ''roberta-base]")
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

total_layers = 12 + 1
results = {i: [] for i in range(total_layers)}
for i in range(total_layers):
    for j in range(total_layers):
        logger.info("Pool:{}, Encoder:{}, Whitening:{}, L:{}, L:{}".format(args.pooling, args.encoder_name, args.whitening, i, j))
        pooling_model = Layer2Pooling(args.pooling, word_embedding_model.get_word_embedding_dimension(), layer_i=i, layer_j=j)

        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

        evaluators = {task: [] for task in target_eval_tasks}  # evaluators has a list of different evaluator classes we call periodically
        sts_reader = STSBenchmarkDataReader(os.path.join(script_folder_path, args.sts_corpus))
        for idx, target in enumerate(target_eval_files):
            output_filename_eval = os.path.join(script_folder_path, args.sts_corpus + target + "-test.csv")
            if args.whitening:
                evaluators[target[:5]].append(WhiteningEmbeddingSimilarityEvaluator.from_input_examples(sts_reader.get_examples(output_filename_eval), measure_data_num=target_eval_data_num[idx], name=target, main_similarity=SimilarityFunction.COSINE))
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
        results[i].append(np.mean(_all_results))

logger.info("***** Finishing evaluation *****")
logger.info("********** Evaluation Results Spearman Cor. **********")
logger.info("\t L0\tL1\tL2\tL3\tL4\tL5\tL6\tL7\tL8\tL9\tL10\tL11\tL12")
for layer_i, results in results.items():
    line = "L{}\t".format(layer_i)
    line += " ".join(['%.3f'%item for item in results])
    logger.info(line)


