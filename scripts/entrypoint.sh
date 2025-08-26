#!/bin/bash

echo "Running in $MODE mode"

if [ "$MODE" == "train" ]; then
    echo "Starting training..."
    exec python3 -u -m src.cli

elif [ "$MODE" = "mlflow" ]; then
    echo "Starting MLflow UI..."
    mlflow ui --backend-store-uri experiments/mlruns --host 0.0.0.0 --port 5001

elif [ "$MODE" == "serve" ]; then
    echo "Serving FastAPI model..."
    uvicorn inference_api.main:app --host 0.0.0.0 --port 8000 --reload

elif [ "$MODE" == "dev" ]; then
    echo "Starting Jupyter Notebook..."
    jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --NotebookApp.token=''

else
    echo "Invalid mode: $MODE"
    exit 1
fi