import getopt
import sys
import torch
import boto3


def get_s3_boto_client(s3_url: str) -> boto3.client:
    return boto3.client("s3", endpoint_url=s3_url)


def download_data_from_s3(s3_client: boto3.client, bucket_name: str, s3_data_path: str, local_root_path: str,
                          local_data_name: str):
    data_path = f"{local_root_path}/{local_data_name}"
    s3_client.download_file(bucket_name, s3_data_path, data_path)


def parse_config_file_path(argv) -> str:
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


def select_hardware_for_training(device_name: str) -> str:
    device_name = device_name.lower()
    if device_name == 'cpu':
        return 'cpu'
    elif device_name == 'gpu':
        if torch.cuda.is_available():
            return 'cuda:0'
        else:
            print("GPU is unavailable on this worker")
            return 'cpu'
    else:
        print("Unknown device name, choose cpu as default device")
        return 'cpu'
