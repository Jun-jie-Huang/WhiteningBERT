import torch
from torch import Tensor
from torch import nn
from typing import Union, Tuple, List, Iterable, Dict
import os
import json
import numpy as np

import logging
logger = logging.getLogger(__name__)


class EasyPooling(nn.Module):

    def __init__(self, pooling, word_embedding_dimension):
        super(EasyPooling, self).__init__()
        self.config_keys = ['word_embedding_dimension']
        self.word_embedding_dimension = word_embedding_dimension
        self.pooling_output_dimension = word_embedding_dimension
        self.pooling = pooling

    def forward(self, features: Dict[str, Tensor]):
        ft_all_layers = features['all_layer_embeddings']  # length 13
        # org_device = ft_all_layers[0].device
        # all_layer_embedding = torch.stack(ft_all_layers).transpose(1,0)  # (Batch, 13, SentLength, 768), 每个batch中SentLength不一样

        first_features = features['token_embeddings']  # (Batch, SentLength, 768)
        # first_features = (ft_all_layers[0] + ft_all_layers[1] + ft_all_layers[2] + ft_all_layers[3] + ft_all_layers[4] +
        #                   ft_all_layers[5] + ft_all_layers[6] + ft_all_layers[7] + ft_all_layers[8] + ft_all_layers[9] +
        #                   ft_all_layers[10] + ft_all_layers[11] + ft_all_layers[12]) / 13

        if self.pooling == 'cls':
            output_vector = features['cls_token_embeddings']
        elif self.pooling == 'aver':
            masks = features['attention_mask']  # (Batch, SentLength)
            first_features = first_features * masks.unsqueeze(2)
            output_vector = torch.sum(first_features, dim=1) / (torch.sum(masks, dim=1, keepdim=True) + 0.01)
        elif self.pooling == 'max':
            output_vector = torch.max(first_features, dim=1)[0]
        elif self.pooling == 'aver_all':
            output_vector = torch.mean(first_features, dim=1)
        else:
            output_vector = features['cls_token_embeddings']

        features.update({'sentence_embedding': output_vector})  # (Batch, 768)

        return features

    def norm_vector(self, vec, p=2, dim=0):
        """
        Implements the normalize() function from sklearn
        """
        vec_norm = torch.norm(vec, p=p, dim=dim)
        return vec.div(vec_norm.expand_as(vec))

    def cosine_similarity_torch(self, x1, x2=None, eps=1e-8):
        x2 = x1 if x2 is None else x2
        w1 = x1.norm(p=2, dim=1, keepdim=True)
        w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
        return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)

    def get_sentence_embedding_dimension(self):
        return self.pooling_output_dimension

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path):
        with open(os.path.join(output_path, 'config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, 'config.json')) as fIn:
            config = json.load(fIn)

        return EasyPooling(**config)


class Layer2Pooling(nn.Module):

    def __init__(self, pooling, word_embedding_dimension, layer_i=0, layer_j=-1):
        super(Layer2Pooling, self).__init__()
        self.config_keys = ['word_embedding_dimension']
        self.word_embedding_dimension = word_embedding_dimension
        self.pooling_output_dimension = word_embedding_dimension
        self.pooling = pooling
        self.layer_i = layer_i
        self.layer_j = layer_j

    def forward(self, features: Dict[str, Tensor]):
        ft_all_layers = features['all_layer_embeddings']  # length 13
        # all_layer_embedding = torch.stack(ft_all_layers)  # (13, Batch, SentLength, 768)
        if self.layer_i != self.layer_j:
            first_features = ft_all_layers[self.layer_i] + ft_all_layers[self.layer_j]
        else:
            first_features = ft_all_layers[self.layer_i]
        # first_features = features['token_embeddings']  # (Batch, SentLength, 768)

        if self.pooling == 'cls':
            # output_vector = features['cls_token_embeddings']
            output_vector = first_features[:, 0, :]
        elif self.pooling == 'aver':
            masks = features['attention_mask']  # (Batch, SentLength)
            first_features = first_features * masks.unsqueeze(2)
            output_vector = torch.sum(first_features, dim=1) / (torch.sum(masks, dim=1, keepdim=True) + 0.01)
        elif self.pooling == 'max':
            output_vector = torch.max(first_features, dim=1)[0]
        elif self.pooling == 'aver_all':
            output_vector = torch.mean(first_features, dim=1)
        else:
            output_vector = features['cls_token_embeddings']

        features.update({'sentence_embedding': output_vector})  # (Batch, 768)

        return features

    def norm_vector(self, vec, p=2, dim=0):
        """
        Implements the normalize() function from sklearn
        """
        vec_norm = torch.norm(vec, p=p, dim=dim)
        return vec.div(vec_norm.expand_as(vec))

    def cosine_similarity_torch(self, x1, x2=None, eps=1e-8):
        x2 = x1 if x2 is None else x2
        w1 = x1.norm(p=2, dim=1, keepdim=True)
        w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
        return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)

    def get_sentence_embedding_dimension(self):
        return self.pooling_output_dimension

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path):
        with open(os.path.join(output_path, 'config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, 'config.json')) as fIn:
            config = json.load(fIn)

        return Layer2Pooling(**config)


class LayerNPooling(nn.Module):

    def __init__(self, pooling, word_embedding_dimension, layers=2):
        super(LayerNPooling, self).__init__()
        self.config_keys = ['word_embedding_dimension']
        self.word_embedding_dimension = word_embedding_dimension
        self.pooling_output_dimension = word_embedding_dimension
        self.pooling = pooling
        self.layers = layers

    def forward(self, features: Dict[str, Tensor]):
        # ft_all_layers = features['all_layer_embeddings']  # length 13
        all_layer_embedding = torch.stack(features['all_layer_embeddings'])  # (13, Batch, SentLength, 768)
        first_features = torch.mean(all_layer_embedding[self.layers, :, :, :], dim=0)
        # logger.info(first_features.shape)
        # first_features = features['token_embeddings']  # (Batch, SentLength, 768)

        if self.pooling == 'cls':
            # output_vector = features['cls_token_embeddings']
            output_vector = first_features[:, 0, :]
        elif self.pooling == 'aver':
            masks = features['attention_mask']  # (Batch, SentLength)
            first_features = first_features * masks.unsqueeze(2)
            output_vector = torch.sum(first_features, dim=1) / (torch.sum(masks, dim=1, keepdim=True) + 0.01)
        elif self.pooling == 'max':
            output_vector = torch.max(first_features, dim=1)[0]
        elif self.pooling == 'aver_all':
            output_vector = torch.mean(first_features, dim=1)
        else:
            output_vector = features['cls_token_embeddings']

        features.update({'sentence_embedding': output_vector})  # (Batch, 768)

        return features

    def norm_vector(self, vec, p=2, dim=0):
        """
        Implements the normalize() function from sklearn
        """
        vec_norm = torch.norm(vec, p=p, dim=dim)
        return vec.div(vec_norm.expand_as(vec))

    def cosine_similarity_torch(self, x1, x2=None, eps=1e-8):
        x2 = x1 if x2 is None else x2
        w1 = x1.norm(p=2, dim=1, keepdim=True)
        w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
        return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)

    def get_sentence_embedding_dimension(self):
        return self.pooling_output_dimension

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path):
        with open(os.path.join(output_path, 'config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, 'config.json')) as fIn:
            config = json.load(fIn)

        return LayerNPooling(**config)

