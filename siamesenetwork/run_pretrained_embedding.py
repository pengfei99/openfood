#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train the siamese network for product description similarity

Application : product description similarity
Architecture : Embeddings Pre-Trained
Loss : Quadruplet loss

Created on Thu Jun 10 2021
@author: jeremylhour, milena-git
"""
import os
import random
from datetime import datetime

import fasttext.util
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from siamesePreTrainedEmbeddings import SiamesePreTrainedQuadruplet
from siameseUtils import AverageMeter, QuadrupletLoss, libel2vec, computeModelTopk, lib2vocab, \
    CreateTorchQuadrupletDataset, vocabtoidx, multipleNWayOneShotTask
from utils import *


# -----------------
# Primary Functions
# -----------------

def train(model, trainDataGenerator, devDataGenerator, output_path, device, n_epochs=10, lr=.0005):
    """ Train the siamese model for single epoch.

    @param model (SiameseNetwork): model for kilometers
    @param trainDataGenerator (): a DataLoader object
    @param devDataGenerator (): a DataLoader object
    @param output_path (str): Path to which model weights and results are written.
    @param device (str): CPU or GPU
    @param n_epochs (int): Number of training epochs
    @param lr (float): Learning rate
    """
    loss_func = QuadrupletLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_dev_loss = float("inf")
    model = model.to(device)  # put the model either on GPU or CPU

    train_loss_history, dev_loss_history = [], []
    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1} out of {n_epochs}")
        train_loss = train_for_epoch(model=model, trainDataGenerator=trainDataGenerator, optimizer=optimizer,
                                     loss_func=loss_func)
        train_loss_history.append(train_loss)

        print(f"Average Train Loss: {round(train_loss, 3)}")
        if device != "cpu":
            torch.cuda.empty_cache()  # clear GPU cache to avoid out of memory, otherwise vectors acumulate

        # evaluate model on dev set
        model.eval()
        loss_meter = AverageMeter()
        for i, data in enumerate(devDataGenerator, 0):
            lib_anchor, lib_pos, lib_neg1, lib_neg2 = data[0], data[1], data[2], data[3]
            output_anchor, output_pos, output_neg1, output_neg2 = model(lib_anchor, lib_pos, lib_neg1, lib_neg2)
            dev_loss = loss_func(output_anchor, output_pos, output_neg1, output_neg2)
            loss_meter.update(dev_loss.item())

        print(f"Average Dev Loss: {round(loss_meter.avg, 3)}")
        dev_loss_history.append(loss_meter.avg)
        if device != "cpu":
            torch.cuda.empty_cache()  # clear GPU cache to avoid out of memory, otherwise vectors acumulate

        # save best model
        if loss_meter.avg < best_dev_loss:
            best_dev_loss = loss_meter.avg
            print("New best dev loss! Saving the model.")
            torch.save(model.state_dict(), output_path)
            # s3.upload_file(output_path, config['bucket'], 'siamese/' + output_path)
        print("")
    return train_loss_history, dev_loss_history


def train_for_epoch(model, trainDataGenerator, optimizer, loss_func):
    """ Train the siamese model for single epoch.

    @param model (SiameseNetwork): model for kilometers
    @param trainDataGenerator (): a DataLoader object
    @param optimizer (nn.Optimizer): Adam Optimizer
    @param loss_func (nn.CrossEntropyLoss): Cross Entropy Loss Function

    @return train_loss (float): loss for train data
    """
    model.train()
    loss_meter = AverageMeter()

    with tqdm(total=(len(trainDataGenerator))) as prog:
        for i, data in enumerate(trainDataGenerator):
            lib_anchor, lib_pos, lib_neg1, lib_neg2 = data[0], data[1], data[2], data[3]

            optimizer.zero_grad()  # remove any baggage in the optimizer

            output_anchor, output_pos, output_neg1, output_neg2 = model(anchor=lib_anchor,
                                                                        positive=lib_pos,
                                                                        negative1=lib_neg1,
                                                                        negative2=lib_neg2)

            loss = loss_func(output_anchor, output_pos, output_neg1, output_neg2)
            loss.backward()
            optimizer.step()
            prog.update(1)
            loss_meter.update(loss.item())

    print("")
    train_loss = loss_meter.avg
    return train_loss


def voc2idx(voc):
    """
    @param voc (list of str):
    """
    voc_dic = dict()
    voc_dic["oov"] = 0
    for i, word in enumerate(voc):
        voc_dic[word] = i + 1
    return voc_dic


def get_params(config_file_path: str):
    print(80 * "=")
    print("LOADING CONFIG")
    print(80 * "=")
    # get config file path
    with open(config_file_path, 'r') as stream:
        config = yaml.safe_load(stream)
    # local data root path on worker
    root_path = config['data_root_path']
    batch_size = config['batch_size']
    n_epochs = config['n_epochs']
    dim = config['dim']
    lr = float(config['lr'])
    freeze_layers = config['freeze_layers']
    print(f'Dimension of the embedding space : {dim}')
    if freeze_layers:
        print('Does not train pre-trained layers.')
    print(f'Batch size : {batch_size}')
    print(f'Learning rate : {lr} \n')
    return root_path, batch_size, n_epochs, dim, lr, freeze_layers


def build_word_embedding_layer(ft_model_path: str, voc_dic: dict):
    print(80 * "=")
    print("LOADING PRE-TRAINED EMBEDDINGS")
    print(80 * "=")

    # load fasttext model to convert french text to vector

    ft = fasttext.load_model(ft_model_path)
    # todo rewrite vocab and embedding

    # CHARGEMENT DES EMBEDDINGS
    matrix_len = len(voc_dic) + 1
    weights_matrix = np.zeros((matrix_len, ft.get_dimension()))
    words_found = 0

    for word in voc_dic:
        try:
            weights_matrix[voc_dic[word]] = ft.get_word_vector(word)
            words_found += 1
        except KeyError:
            weights_matrix[voc_dic[word]] = np.random.normal(scale=0.6, size=(ft.get_dimension(),))

    print(f'We have found {words_found} in our fastText Pretrained Model.')
    print(weights_matrix.shape)
    return torch.FloatTensor(weights_matrix)


def build_train_dev_data_loader(train_data_path: str, dev_data_path: str, voc_dic: dict, device: str, batch_size: int):
    print(80 * "=")
    print("LOADING AND PRE-PROCESSING TRAINING DATA")
    print(80 * "=")

    # load training data
    libTrainData = CreateTorchQuadrupletDataset(voc_dic, train_data_path, device=device)
    print(f'Train dataset : {len(libTrainData)} quadruplets of descriptions loaded. \n')

    print(80 * "=")
    print("LOADING AND PRE-PROCESSING DEV DATA")
    print(80 * "=")

    # load dev/validation data
    libDevData = CreateTorchQuadrupletDataset(voc_dic, dev_data_path, device=device)
    print(f'Dev dataset : {len(libDevData)} quadruplets of descriptions loaded. \n')

    # build dataloader
    train_data_loader = DataLoader(libTrainData, shuffle=True, batch_size=batch_size)
    dev_data_loader = DataLoader(libDevData, shuffle=True, batch_size=batch_size)

    return train_data_loader, dev_data_loader


def main(argv):
    print('\nThis is a script for training a siamese network using fastText embeddings with a Quadruplet Loss.\n')
    # Step1: Get param from config file
    config_file = parse_config_file_path(argv)
    # hyper parameter to be logged for each run:
    # - batch_size
    # - n_epochs
    # - dim
    # - lr: learning_rate

    # other param:
    # freeze_layers: layers don't need to be trained
    # root_path: local root path for downloaded data on each worker
    root_path, batch_size, n_epochs, dim, lr, freeze_layers = get_params(config_file)

    device = select_hardware_for_training('gpu')

    # step2 : load pretrained fasttext model and build a word embedding layer
    # get fasttext embedding model path
    ft_model_path = f"{root_path}/models_pretrained/coicop/model_compressed.ftz"
    # get vocabulary local path
    vocab_path = f"{root_path}/vocab.txt"
    # build a mapping between word and an index, need to be rewrite
    voc_dic = vocabtoidx(vocab=vocab_path)
    weights_matrix = build_word_embedding_layer(ft_model_path, voc_dic)

    # step3 : transfer data
    train_data_path = f"{root_path}/train.csv"
    dev_data_path = f"{root_path}/dev.csv"
    train_data_loader, dev_data_loader = build_train_dev_data_loader(train_data_path, dev_data_path, voc_dic, device,
                                                                     batch_size)

    print(80 * "=")
    print("TRAINING THE MODEL")
    print(80 * "=")

    # step4: build and init model
    siameseModel = SiamesePreTrainedQuadruplet(weights_matrix=weights_matrix, dim=dim, length=20)

    train_from_previous = False
    if train_from_previous:
        siameseModel.load_state_dict(torch.load('models_results/20210616_152847/model.weights'))

    # output_path
    output_dir = "{}/models_results/{:%Y%m%d_%H%M%S}/".format(root_path, datetime.now())
    output_path = output_dir + "model.weights"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # step5: train the model
    train_loss_history, dev_loss_history = train(model=siameseModel,
                                                 trainDataGenerator=train_data_loader,
                                                 devDataGenerator=dev_data_loader,
                                                 output_path=output_path,
                                                 device=device,
                                                 n_epochs=n_epochs,
                                                 lr=lr)

    plt.plot(np.arange(1, len(train_loss_history) + 1), train_loss_history, 'b')
    plt.plot(np.arange(1, len(dev_loss_history) + 1), dev_loss_history, 'r')
    plt.show()

    # step6: evaluate model by using test data set
    print(80 * "=")
    print("COMPUTING VECTORS FOR TEST SET")
    print(80 * "=")

    testTheBest = True
    if testTheBest:
        siameseModel.load_state_dict(torch.load(output_path))
        print('Testing the best model over training epochs.\n')

    # load test data
    test_data_path = f"{root_path}/test.csv"
    df = pd.read_csv(test_data_path, nrows=2000)
    list_libel = lib2vocab(df['libel_clean'].to_list(), voc_dic)
    list_libel_OFF = lib2vocab(df['libel_clean_OFF'].to_list(), voc_dic)

    list_libel = [torch.tensor([l]) for l in list_libel]
    list_libel_OFF = [torch.tensor([l]) for l in list_libel_OFF]

    vec_libel, libel, idx_to_libel = libel2vec(siameseModel.to('cpu'), list_libel)
    vec_OFF, libel_OFF, idx_to_libel_OFF = libel2vec(siameseModel.to('cpu'), list_libel_OFF)

    print(80 * "=")
    print("COMPUTING MODEL'S TOP-K ACCURACY")
    print(80 * "=")

    accuracy, outliers = computeModelTopk(target_vectors=vec_libel,
                                          vectors=vec_OFF,
                                          target_labels=df['libel_clean'].to_list(),
                                          labels=df['libel_clean_OFF'].to_list(),
                                          K=10)

    with open(output_dir + 'performance.txt', 'a') as f:
        for i, item in enumerate(accuracy):
            f.write('Average Test Accuracy for top-{} : {:.2f}\n'.format(i + 1, item))

    # correction = False
    # if correction:
    #     print(80 * "=")
    #     print("TESTING THE MODEL USING N-WAY ONE SHOT TASKS")
    #     print(80 * "=")
    #
    #     N_maxParam = 10
    #
    #     gotItPercentage = multipleNWayOneShotTask(vectors=watch_vectors, labels=family, N_max=N_maxParam, B=500)
    #
    #     plt.plot(np.arange(2, N_maxParam + 1), gotItPercentage, 'b')
    #     plt.plot(np.arange(2, N_maxParam + 1), 1 / np.arange(2, N_maxParam + 1), 'r')
    #     plt.show()
    #
    #     with open(output_dir + 'performance.txt', 'a') as f:
    #         for i, item in enumerate(gotItPercentage):
    #             f.write('Average Test Accuracy for {}-way task : {:.2f}\n'.format(i + 2, item))


if __name__ == "__main__":
    main(sys.argv[1:])
