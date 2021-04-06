# WhiteningBERT

Source code and data for paper [WhiteningBERT: An Easy Unsupervised Sentence Embedding Approach](https://arxiv.org/abs/2104.01767).

## Preparation

```
git clone https://github.com/Jun-jie-Huang/WhiteningBERT.git
pip install -r requirements.txt
cd examples/evaluation
```

## Usage

#### Datasets 

We use seven STS datasets, including STSBenchmark, SICK-Relatedness, STS12, STS13, STS14, STS15, STS16.

The processed data can be found in [./examples/datasets/](./examples/datasets/). 

#### Run

1. To run a quick demo:

```
python evaluation_stsbenchmark.py \
			--pooling aver \
			--layer_num 1,12 \
			--whitening \
			--encoder_name bert-base-cased
```

Specify `--pooing` with `cls` or `aver` to choose whether use the [CLS] token or averaging all tokens. Also specify `--layer_num` to combine layers, separated by a comma.

2. To enumerate all possible combinations of two layers and automatically evaluate the combinations consequently:

```
python evaluation_stsbenchmark_layer2.py \
			--pooling aver \
			--whitening \
			--encoder_name bert-base-cased
```

3. To enumerate all possible combinations of N layers:

```
python evaluation_stsbenchmark_layerN.py \
			--pooling aver \
			--whitening \
			--encoder_name bert-base-cased\
			--combination_num 4
```

4. You can also save the embeddings of the sentences

```
python evaluation_stsbenchmark_save_embed.py \
			--pooling aver \
			--layer_num 1,12 \
			--whitening \
			--encoder_name bert-base-cased \
			--summary_dir ./save_embeddings
```

#### A list of PLMs you can select: 

- `bert-base-uncased` , ` bert-large-uncased `
- `roberta-base`, `roberta-large `
- `bert-base-multilingual-uncased`
- `sentence-transformers/LaBSE`
- `albert-base-v1 `, `albert-large-v1 ` 
- `microsoft/layoutlm-base-uncased `, `microsoft/layoutlm-large-uncased `
- `SpanBERT/spanbert-base-cased `, `SpanBERT/spanbert-large-cased `
- `microsoft/deberta-base `, `microsoft/deberta-large `
- `google/electra-base-discriminator`
- `google/mobilebert-uncased `
- `microsoft/DialogRPT-human-vs-rand `
- `distilbert-base-uncased`
- ......

## Acknowledgements

Codes are adapted from the repos of the EMNLP19 paper [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://github.com/UKPLab/sentence-transformers) and the EMNLP20 paper [An Unsupervised Sentence Embedding Method by Mutual Information Maximization](https://github.com/yanzhangnlp/IS-BERT/)