#!/bin/bash

export MLFLOW_S3_ENDPOINT_URL=https://minio.lab.sspcloud.fr
export AWS_ACCESS_KEY_ID=changeMe
export AWS_SECRET_ACCESS_KEY=changeMe
export AWS_SESSION_TOKEN=changeMe
export AWS_DEFAULT_REGION=us-east-1

export MLFLOW_TRACKING_URI='changeMe'

python /home/pliu/git/openfood/siamesenetwork/fetch_model.py