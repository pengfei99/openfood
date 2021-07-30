import math
import os
import random
import zipfile

import pandas as pd
import yaml

from utils import *


def split_data(seed: int, root_path: str):
    random.seed(seed)
    df = pd.read_csv("{}/data_siamese.csv".format(root_path))
    test_ids = random.sample(list(df.index), math.floor(df.shape[0] * 0.2))
    df_test = df.loc[test_ids]
    df_train = df.loc[~df.index.isin(test_ids)]
    dev_ids = random.sample(list(df_train.index), math.floor(df_train.shape[0] * 0.1))
    df_dev = df_train.loc[dev_ids]
    df_train = df_train.loc[~df_train.index.isin(dev_ids)]
    df_train[['libel_clean', 'libel_clean_OFF']].to_csv('{}/train.csv'.format(root_path))
    df_dev[['libel_clean', 'libel_clean_OFF']].to_csv('{}/dev.csv'.format(root_path))
    df_test[['libel_clean', 'libel_clean_OFF']].to_csv('{}/test.csv'.format(root_path))
    return df_train, df_dev, df_test


def download_model_result_from_s3(s3_client, config, root_path):
    model_results_path = f"{root_path}/models_results"
    last_model_results_path = f"{root_path}/models_results/{config['lastModel']}"
    last_model_weights_path = f"{last_model_results_path}/model.weights"

    if not os.path.exists(last_model_results_path):
        if not os.path.exists(model_results_path):
            os.mkdir(model_results_path)
        os.mkdir(last_model_results_path)
    s3_client.download_file(config['bucket'], config['modelKey'] % config['lastModel'], last_model_weights_path)


def download_pretrained_model_from_s3(s3_client, config: dict, root_path: str):
    pretrained_model_path = f"{root_path}/models_pretrained"
    coicop_model_path = f"{pretrained_model_path}/{config['pretrainedModel_COICOP']}"
    na2008_model_path = f"{pretrained_model_path}/{config['pretrainedModel_NA2008']}"
    extracted_coicop_model_path = f"{pretrained_model_path}/coicop"
    extracted_na2008_model_path = f"{pretrained_model_path}/NA2008"
    if not os.path.exists(pretrained_model_path):
        os.mkdir(pretrained_model_path)
        s3_client.download_file(config['bucket'], config['fasttextmodelKey'] % config['pretrainedModel_COICOP'],
                                coicop_model_path)
        s3_client.download_file(config['bucket'], config['fasttextmodelKey'] % config['pretrainedModel_NA2008'],
                                na2008_model_path)

    with zipfile.ZipFile(coicop_model_path, 'r') as extractor:
        extractor.extractall(extracted_coicop_model_path)
    with zipfile.ZipFile(na2008_model_path, 'r') as extractor:
        extractor.extractall(extracted_na2008_model_path)


def main(argv):
    # setup config
    config_file_path = parse_config_file_path(argv)
    seed = 5648783
    root_path = "/home/jovyan/work/openfood_data"
    with open(config_file_path, 'r') as stream:
        config = yaml.safe_load(stream)
    s3_client = get_s3_boto_client(config['s3url'])

    # Step1: download data from
    local_data_name = "data_siamese.csv"
    download_data_from_s3(s3_client, config['bucket'], config['dataKey'], root_path, local_data_name)

    # Step2: Split data to train, test, validation
    split_data(seed, root_path)

    # Step3: download models
    download_model_result_from_s3(s3_client, config, root_path)
    download_pretrained_model_from_s3(s3_client, config, root_path)

    # Step4: download vocab
    local_vocab_name = "vocab.txt"
    download_data_from_s3(s3_client, config['bucket'], config['vocab'], root_path, local_vocab_name)


if __name__ == "__main__":
    main(sys.argv[1:])
