# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Define the siamese network for one-shot learning,
for french short labels
02/06/2021
@author: milena-git, from jeremylhour courtesy
"""
import torch
import torch.nn as nn


def _createEmbeddingLayer(weights_matrix, non_trainable=False):
    """
    _createEmbeddingLayer:
        create a layer from pre-trained embeddings

    @param weights_matrix (np.array):
    @param non_trainable (bool):
    """
    weights_matrix = torch.tensor(weights_matrix)
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False
    return emb_layer, num_embeddings, embedding_dim


class SiamesePreTrainedQuadruplet(nn.Module):

    def __init__(self, weights_matrix, length, dim=100):
        """
        Initialize the siamese network with pre-trained embeddings

        @param weights_matrix (torch.tensor):
        @param length (int): longueur des inputs
        @param dim (int): dimension of the output embedding space
        """
        super(SiamesePreTrainedQuadruplet, self).__init__()
        self.dim = dim
        self.length = length
        self.embedding = nn.Embedding.from_pretrained(weights_matrix, padding_idx=0)
        self.fc1 = nn.Sequential(
            nn.Linear(self.length * weights_matrix.size()[1], 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 800),
            nn.Dropout(0.2),
            nn.Linear(800, 500),
            nn.Dropout(0.2),
            nn.Linear(500, self.dim)
        )

    def forward_once(self, x):
        """
        Run one of the network on a single image

        @param x (): img output from SiameseNetworkDataset
        """
        embedded = self.embedding(x)
        embedded = torch.reshape(embedded, (embedded.size()[0], embedded.size()[1] * embedded.size()[2]))
        output = self.fc1(embedded)
        return output

    def forward(self, anchor, positive, negative1, negative2):
        """
        Run the model forward, by applying forward_once to each inputs
        Main forward that is used during train, wraps forward_once().

        @param anchor, positive, negative1, negative2 (): output from SiameseNetworkDataset
        """
        anchor_o, positive_o, negative1_o, negative2_o = self.forward_once(anchor), self.forward_once(
            positive), self.forward_once(negative1), self.forward_once(negative2)
        return anchor_o, positive_o, negative1_o, negative2_o


if __name__ == '__main__':
    pass
