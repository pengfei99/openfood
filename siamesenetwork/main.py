import math
import os
import random
import zipfile

import pandas as pd
import yaml
import getopt
import sys

from utils import get_s3_boto_client, download_data_from_s3


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


def input_parser(argv) -> str:
    config_file = None
    hint = "main.py -c <config_file>"
    try:
        # hc: is the short option definitions. For example, you can test.py -c or test.py -h
        # [cfile,help] for long option definitions. For example, you can do test.py --cfile or test.py --help
        opts, args = getopt.getopt(argv, "hc:", ["cfile=", "help="])
    except getopt.GetoptError:
        raise SystemExit(f"invalide arguments \nhint: {hint}")
    for opt, arg in opts:
        # option h for help
        if opt in ('-h', "--help"):
            print("hint")
            sys.exit()
        # option for config file
        elif opt in ("-c", "--cfile"):
            config_file = arg
        else:
            print("unknown option.\n " + hint)
    if (not opts) or len(args) > 1:
        raise SystemExit(f"invalide arguments \nhint: {hint}")
    print(f'Config file path is {config_file}')
    return config_file


def main(argv):
    # setup config
    config_file_path = input_parser(argv)
    seed = 5648783

    with open(config_file_path, 'r') as stream:
        config = yaml.safe_load(stream)
    # get root path for local data
    root_path = config['data_root_path']
    if not os.path.exists(root_path):
        os.mkdir(root_path)

    # create s3 client
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
