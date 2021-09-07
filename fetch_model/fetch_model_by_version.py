import logging
import os

import mlflow
import pandas as pd
import torch

from siamesenetwork.siameseUtils import lib2vocab, vocabtoidx, libel2vec, computeModelTopk

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


# In this example, We will fetch a model based on its version, when you add a run to model repo, you can choose a
# specific name, if empty, the default numeric name will be given which starts from 1.

def prepare_data(df, voc_dic):
    list_libel = lib2vocab(df['libel_clean'].to_list(), voc_dic)
    list_libel_off = lib2vocab(df['libel_clean_OFF'].to_list(), voc_dic)
    list_libel_tensor = [torch.tensor([item]) for item in list_libel]
    list_libel_off_tensor = [torch.tensor([item]) for item in list_libel_off]
    return list_libel_tensor, list_libel_off_tensor


def get_vocab(vocab_path: str):
    return vocabtoidx(vocab=vocab_path)


def load_model_by_version(model_name: str, version: str):
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{version}")
    return model


def predict(model, libel_tensor, libel_off_tensor, df):
    vec_libel, libel, idx_to_libel = libel2vec(model.to('cpu'), libel_tensor)
    vec_off, libel_off, idx_to_libel_off = libel2vec(model.to('cpu'), libel_off_tensor)
    k = 10
    accuracy, outliers = computeModelTopk(target_vectors=vec_libel,
                                          vectors=vec_off,
                                          target_labels=df['libel_clean'].to_list(),
                                          labels=df['libel_clean_OFF'].to_list(),
                                          K=k)
    for i, item in enumerate(accuracy):
        print('Average Test Accuracy for top-{} : {:.2f}\n'.format(i + 1, item))


def main():
    # set up the ml server url and experiment name
    remote_server_uri = "https://pengfei-mlflow-4631160170256180972-mlflow-ihm.kub.sspcloud.fr"
    os.environ["MLFLOW_TRACKING_URI"] = remote_server_uri

    # step1: prepare data
    vocab_path = "../data/vocab.txt"
    test_data_path = "../data/test.csv"
    vocab = get_vocab(vocab_path)
    df = pd.read_csv(test_data_path)
    libel_tensor, libel_off_tensor = prepare_data(df, vocab)

    # step2: download model from model repo
    model_name = "openfood"
    version = "2"
    model = load_model_by_version(model_name, version)

    # step3: predict on
    predict(model, libel_tensor, libel_off_tensor, df)


if __name__ == "__main__":
    main()
