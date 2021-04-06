# -*- coding: UTF-8 -*-
import os, sys
import collections
import csv
import json
import random
import logging
import pickle
import re
import shutil

import numpy as np
import torch

from sklearn.metrics import f1_score, precision_score, recall_score
from scipy.stats import pearsonr, spearmanr, rankdata


logger = logging.getLogger(__name__)
csv.field_size_limit(sys.maxsize)


def PCA_svd(X, k, center=True):
    n = X.size()[0]
    ones = torch.ones(n).view([n,1])
    h = ((1/n) * torch.mm(ones, ones.t())) if center else torch.zeros(n*n).view([n,n])
    H = torch.eye(n) - h
    H = H.to(X.device)
    X_center = torch.mm(H.double(), X.double())
    u, s, v = torch.svd(X_center)
    # components = u[:, :k]
    # explained_variance = torch.mul(s[:k], s[:k])/(n-1)
    components = v[:k]

    # W = torch.matmul(u, torch.diag(1/torch.sqrt(s)))
    # components = torch.matmul(X.double().transpose(0,1), W)
    return components


def whitening_torch_final(embeddings):
    mu = torch.mean(embeddings, dim=0, keepdim=True)
    cov = torch.mm((embeddings - mu).t(), embeddings - mu)
    u, s, vt = torch.svd(cov)
    W = torch.mm(u, torch.diag(1/torch.sqrt(s)))
    embeddings = torch.mm(embeddings - mu, W)
    return embeddings


def whitening_pca(embedding1, embedding2, k=768):
    num = embedding1.size()[0]
    transformed_embedding = PCA_svd(torch.cat([embedding1, embedding2], dim=0).t(), num*2)
    # logger.info(torch.cat([embedding1, embedding2], dim=0).size())
    # logger.info(transformed_embedding.size())
    # logger.info("whitening pca")
    return transformed_embedding[:num, :], transformed_embedding[num:, :]


def transform_and_normalize(vecs, kernel=None, bias=None):
    """应用变换，然后标准化
    """
    if not (kernel is None or bias is None):
        # vecs = (vecs + bias).dot(kernel) - bias
        vecs = (vecs + bias).dot(kernel)
        # vecs = vecs.dot(kernel)
        # vecs = vecs + bias
    # return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5
    # logger.info('+++bisa')
    return vecs


def transform_and_normalize_torch(vecs, kernel=None, bias=None):
    if not (kernel is None or bias is None):
        vecs = torch.mm(vecs + bias, kernel)
    return vecs / torch.sum(vecs**2, dim=1, keepdim=True)**0.5

