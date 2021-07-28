#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
All useful functions for running siamese networks, including loss functions
Created on Wed Feb 17 20:54:23 2021
@author: jeremylhour, updated milena-git (for textual data)
"""
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


# -----------------
# Simple utils
# -----------------

class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CreateQuadrupletDataset(Dataset):

    def __init__(self, csvpath='datasiamese.csv', transform=None):
        """
        Initializes the CreateQuadrupletDataset

        @param transform (transforms.Compose):
        """
        self.data = pd.read_csv(csvpath)
        self.transform = transform

    def __getitem__(self, index):
        """
        __getitem__:
            method to get a four libels:
                - an anchor,
                - a positive example,
                - a negative example,
                - a negative example also different from the other negative.

        @return the four libels
        """

        randomId = random.choice(self.data.index)
        anchor = self.data.loc[randomId, 'libel_clean']
        positive = self.data.loc[randomId, 'libel_clean_OFF']

        while True:
            # keep looping until a different libel is found
            randomIdAlt = random.choice(self.data.index)
            negative1 = self.data.loc[randomIdAlt, 'libel_clean_OFF']
            if randomIdAlt != randomId:
                break
        while True:
            # keep looping until a different libel is found
            randomIdAlt2 = random.choice(self.data.index)
            negative2 = self.data.loc[randomIdAlt2, 'libel_clean_OFF']
            if randomIdAlt != randomIdAlt2:
                break

        if self.transform is not None:
            anchor, positive, negative1, negative2 = self.transform(anchor), self.transform(positive), self.transform(
                negative1), self.transform(negative2)

        return [anchor, positive, negative1, negative2]

    def __len__(self):
        return self.data.shape[0]


class SiameseNetworkDataset(Dataset):

    def __init__(self, csvpath='datasiamese.csv', transform=None, ):
        """
        Initializes the SiameseNetworkDataset

        @param csvpath: path to csv data
        @param transform
        """
        self.data = pd.read_csv(csvpath)
        self.transform = transform

    def __getitem__(self, index):
        """
        __getitem__:
            method to get a pair of labels, either matched through EAN or from two different matches

        @return libel0, libel1 and 1 if they are the same, 0 otherwise
        """
        randomId = random.choice(self.data.index)
        libel0 = self.data.loc[randomId, 'libel_clean']

        should_get_same_class = random.randint(0, 1)  # draw at random either a same class image or another class image

        if should_get_same_class:
            libel1 = self.data.loc[randomId, 'libel_clean_OFF']
        else:
            while True:
                # keep looping until a different libel is found
                randomIdAlt = random.choice(self.data.index)
                libel1 = self.data.loc[randomIdAlt, 'libel_clean_OFF']
                if randomIdAlt != randomId:
                    break

        if self.transform is not None:
            libel0, libel1 = self.transform(libel0), self.transform(libel1)

        return libel0, libel1

    def __len__(self):
        return self.data.shape[0]


# -----------------
# Quadruplet Loss
# -----------------

class QuadrupletLoss(torch.nn.Module):
    """
    QuadrupletLoss:
        Implements the quadruplet loss function.
        I divide it by 2 so that it's on the same scale as the triplet loss
    """

    def __init__(self, margin1=2.0, margin2=1.0):
        """
        Initializes the Quadruplet Loss Function.

        @param margin1 (float): margin for the first term
        @param margin2 (float): margin for the second term. Must be smaller than margin1.
        """
        super(QuadrupletLoss, self).__init__()
        self.margin1 = margin1
        self.margin2 = margin2

    def forward(self, anchor, positive, negative1, negative2):
        """
        forward:

        @param anchor: anchor image
        @param positive: image of the same class as anchor
        @param negative1: image of a different class as negative1
        @param negative2: image of a different class as anchor and negative1
        """
        dist_anchor_to_positive = torch.pow(F.pairwise_distance(anchor, positive, keepdim=True), 2)
        dist_anchor_to_negative1 = torch.pow(F.pairwise_distance(anchor, negative1, keepdim=True), 2)
        dist_negative1_to_negative2 = torch.pow(F.pairwise_distance(negative1, negative2, keepdim=True), 2)

        loss_quadruplet = F.relu(dist_anchor_to_positive - dist_anchor_to_negative1 + self.margin1) + F.relu(
            dist_anchor_to_positive - dist_negative1_to_negative2 + self.margin2)
        return loss_quadruplet.mean() / 2.0


# -------------------------------------------
# Compute vector representation of libels
# -------------------------------------------

def libel2vec(model, libels):
    """
    libel2vec:
        computes the np.array of vectors from a siamese model for libels in list_libels

    @param model (): a siamese model that allows a forward_once() method
    @param libels: list of libels
    """

    lib2vec, labels = [], []
    idx_to_label = {}
    i = 0

    for libel in libels:
        output = model.forward_once(libel)
        lib2vec.append(output.cpu().detach().numpy())
        labels.append(libel)
        if labels[i] in idx_to_label.keys():
            idx_to_label[labels[i]] = idx_to_label[labels[i]] + [i]
        else:
            idx_to_label[labels[i]] = [i]
        i += 1

    lib2vec = np.array(lib2vec)

    return np.squeeze(lib2vec), labels, idx_to_label


# ---------------------------------
# Test on N-ways One Shot Tasks
# ---------------------------------

# def runNWayOneShotTask(model, dataset_dir, resolution, N_max=5):
#     """
#     runNWayOneShotTask:
#         Computes accuracy for N-ways one shot learning tasks up to N_max ways
#
#     @param model (siameseNetwork): a siamese model
#     @param dataset_dir (str): directory of picture dataset
#     @param resolution (int): resolution of the picture
#     @param N_max (int): Running N-ways one shot tasks up to N=N_max
#     """
#     model.eval()  # places model in eval modes
#
#     testData = dset.ImageFolder(root=dataset_dir)
#     testDataLoader = CreateOneShotDataset(imageFolderDataset=testData,
#                                           transform=transforms.Compose([transforms.Resize((resolution, resolution)),
#                                                                         transforms.ToTensor()
#                                                                         ]),
#                                           B=N_max + 1)
#     test_dataloader = DataLoader(testDataLoader, num_workers=0, batch_size=1, shuffle=True)
#
#     modelGotItOverNTasks = []
#     for i, data in enumerate(test_dataloader, 0):
#         outputs, distances = [], []
#         for img in data:
#             output = model.forward_once(img)
#             outputs.append(output)
#             distance = F.pairwise_distance(outputs[0], output)
#             distances.append(distance.item())
#
#         distances.pop(0)  # get rid of the first distance : it's from anchor to itself, so it's zero
#         modelGotIt = []
#         for B in range(2, N_max + 1):
#             modelGotIt.append(int(np.argmin(distances[0:B]) == 0))
#         modelGotItOverNTasks.append(modelGotIt)
#
#     # compute mean and print
#     modelGotItOverNTasks = np.array(modelGotItOverNTasks)
#     accuracy = modelGotItOverNTasks.mean(axis=0)
#     for i, item in enumerate(accuracy):
#         print('Average Test Accuracy for {}-way task : {:.2f}'.format(i + 2, item))
#     return accuracy
#

# BELOW : efficient implementation of one-shot task
def _efficientNWayOneShotTask(vectors, labels, N_max):
    """
    _efficientNWayOneShotTask:
        Efficiently implements N-way one shot tasks by sampling vectors.

    @param vectors (np.array): watch vectors, output of siamese model
    @param labels (list of int): true label of the watch
    @param N_max (int): Running N-ways one shot tasks up to N=N_max
    """
    modelGotItOverNTasks = []
    for anchor_idx, anchor in enumerate(vectors):
        other_indices = []

        # draw one positive example
        while True:
            positive_idx = random.choice(range(len(vectors)))
            if labels[positive_idx] == labels[anchor_idx] and positive_idx != anchor_idx:
                other_indices.append(positive_idx)
                break

        # draw N_max negatives examples
        b = 0
        while b < N_max:
            negative_idx = random.choice(range(len(vectors)))
            if labels[negative_idx] != labels[anchor_idx]:
                other_indices.append(negative_idx)
                b += 1

                # compute all distances
        distances = np.sum((vectors[other_indices] - anchor) ** 2, axis=1)

        modelGotIt = []
        for B in range(2, N_max + 1):
            modelGotIt.append(int(np.argmin(distances[0:B]) == 0))
        modelGotItOverNTasks.append(modelGotIt)

    # compute mean and print
    modelGotItOverNTasks = np.array(modelGotItOverNTasks)
    accuracy = modelGotItOverNTasks.mean(axis=0)
    return accuracy


def multipleNWayOneShotTask(vectors, labels, N_max, B=100):
    """
    multipleNWayOneShotTask:
        Efficiently implements N-way one shot tasks by sampling vectors x B.
        (Monte-Carlo)

    @param vectors (np.array): watch vectors, output of siamese model
    @param labels (list of int): true label of the watch
    @param N_max (int): Running N-ways one shot tasks up to N=N_max
    @param B (int): number of simulations
    """
    results = []
    for b in range(B):
        accuracy = _efficientNWayOneShotTask(vectors, labels, N_max)
        results.append(accuracy)

    # compute mean and print
    QUANTILE = 1.96
    results = np.array(results)
    accuracy_mean = results.mean(axis=0)
    accuracy_sd = results.std(axis=0)
    for i, item in enumerate(accuracy_mean):
        print('Average Test Accuracy for {}-way task : {:.2f}'.format(i + 2, item))
        print('.95 confidence Interval : [{:.2f} - {:.2f}]'.format(item - QUANTILE * accuracy_sd[i] / np.sqrt(B),
                                                                   item + QUANTILE * accuracy_sd[i] / np.sqrt(B)))
        print('')
    return accuracy_mean


# -----------------
# Top-k
# -----------------

def _topk(target_vec, vectors, k=1):
    """
    _topk:
        compute top-k closest vector from target_vec in vectors

    @param target_vec (np.array): vector of shape n_dim
    @param vectors (np.array): np.array of shape (n_vectors, n_dim)
    @param k (int): the k in top-k

    @return indices of the top k nearest to target_vec amongst vectors,
        may include the vector itself as first if target_vec is inside the vectors
    """
    distances = np.sum((vectors - target_vec) ** 2, axis=1)
    topk = distances.argsort()[:k]
    return topk


def computeModelTopk(target_vectors, vectors, target_labels, labels, K=10, printMatching=False):
    """
    computeModelTopk:
        computes top-k accuracy for the given vectors

    @param vectors (np.array): np.array of shape (n_vectors, n_dim)
    @param labels (list): list of true labels
    @param K (int): the k in top-k
    """
    top_k_for_vecs = []

    for i, vec in enumerate(target_vectors):
        top_idx = _topk(vec, vectors, k=K)
        is_in_top_k = []

        for k in range(K):

            if k == 2 and printMatching == True:
                print(target_labels[i] + ' is closest to:')
                print([labels[j] for j in top_idx[:(k + 1)]])
                print('The matched pair to be found was:')
                print(i, labels[i])
                print(80 * "=")

            if labels[i] in [labels[j] for j in top_idx[:(k + 1)]]:
                is_in_top_k.append(1)
            else:
                is_in_top_k.append(0)
        top_k_for_vecs.append(is_in_top_k)

    # compute mean and print
    top_k_for_vecs = np.array(top_k_for_vecs)
    accuracy = top_k_for_vecs.mean(axis=0)

    for k, item in enumerate(accuracy):
        print('Top-{} Accuracy : {:.2f}'.format(k + 1, item))

    # get outliers (can't be predicted in last top-k)
    is_outlier = [item[-1] == 0 for item in top_k_for_vecs]
    outliers = np.where(is_outlier)

    return accuracy, outliers[0].tolist()


# ---- libel to index vocab

def vocabtoidx(vocab="vocab.txt"):
    """
    Dictionnary word to idx
    @param: "vocab.txt" text file with words vocabulary (one token per line)
    """
    with open(vocab, 'r') as f:
        vcb = f.readlines()
    vcb = [x.strip() for x in vcb]

    vocid = dict()
    for i, word in enumerate(vcb):
        vocid[word] = i + 1
    return vocid


def lib2vocab(libels, vocab, padding_idx=0, length=20):
    """
    transforms a list of strings to a list of several integers following the vocab dict

    @param libels (lst of str): list of libelles str
    @param vocab (dict): output of vocabtoidx func
    """
    libelsidx = []
    for lib in libels:
        sized_lib = lib.split()[0:length]
        sized_lib += [padding_idx] * (length - len(sized_lib))
        libelsidx.append([vocab[word] if word in vocab.keys() else padding_idx for word in sized_lib])

    return libelsidx


class CreateTorchQuadrupletDataset(Dataset):

    def __init__(self, vocab_dic, csvpath='datasiamese.csv', device="cpu", transform=None):
        """
        Initializes the CreateQuadrupletDataset

        @param transform (transforms.Compose):
        """
        self.data = pd.read_csv(csvpath)
        self.transform = transform
        self.vocab = vocab_dic
        self.device = device

    def __getitem__(self, index):
        """
        __getitem__:
            method to get a four libels:
                - an anchor,
                - a positive example,
                - a negative example,
                - a negative example also different from the other negative.

        @return the four libels
        """

        randomId = random.choice(self.data.index)
        anchor = self.data.loc[randomId, 'libel_clean']
        positive = self.data.loc[randomId, 'libel_clean_OFF']

        while True:
            # keep looping until a different libel is found
            randomIdAlt = random.choice(self.data.index)
            negative1 = self.data.loc[randomIdAlt, 'libel_clean_OFF']
            if randomIdAlt != randomId:
                break
        while True:
            # keep looping until a different libel is found
            randomIdAlt2 = random.choice(self.data.index)
            negative2 = self.data.loc[randomIdAlt2, 'libel_clean_OFF']
            if randomIdAlt != randomIdAlt2:
                break

        if self.transform is not None:
            anchor, positive, negative1, negative2 = self.transform(anchor), self.transform(positive), self.transform(
                negative1), self.transform(negative2)

        anchor, positive, negative1, negative2 = lib2vocab([anchor, positive, negative1, negative2], self.vocab)
        anchor, positive, negative1, negative2 = torch.tensor(anchor), torch.tensor(positive), torch.tensor(
            negative1), torch.tensor(negative2)
        anchor, positive, negative1, negative2 = anchor.to(self.device), positive.to(self.device), negative1.to(
            self.device), negative2.to(self.device)

        return anchor, positive, negative1, negative2

    def __len__(self):
        return self.data.shape[0]