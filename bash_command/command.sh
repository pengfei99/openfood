pip install -r requirements.txt

export MLFLOW_TRACKING_URI='https://pengfei-mlflow-5806178270327072668-mlflow-ihm.kub.sspcloud.fr'
export MLFLOW_EXPERIMENT_NAME="openfood"
export MLFLOW_S3_ENDPOINT_URL=https://minio.lab.sspcloud.fr

python run_pretrained_embedding_with_mlflow.py -c config.yml