#!/bin/bash

export MLFLOW_TRACKING_URI='https://pengfei-mlflow-6923080064718463498-mlflow-ihm.kub.sspcloud.fr'
export MLFLOW_EXPERIMENT_NAME="openfood"
export MLFLOW_S3_ENDPOINT_URL=https://minio.lab.sspcloud.fr

# optional -n <number of epochs> -l <learning rate>
python run_pretrained_embedding_with_mlflow.py -c config.yml