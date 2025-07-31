This project was initially inspired by Daniel Bourke's PyTorch Deep Learning series (https://github.com/mrdbourke/pytorch-deep-learning.git). While the overall system, pipeline design, and MLOps integrations are original and significantly expanded, one early notebook in this repository was adapted from concepts and patterns explored in his tutorials.

It is impossible to design an entire system in your head. Many functions and decisions regarding implementation are made during the course of the project, so it is worth starting from the ground up. In this course we will learn how to create and design AI systems in three levels of complexity:

* Level 1 (foundational pipeline):
  - The skeleton of AI system - how system works?
  - Unit tests and simple integration tests
  - Containerization logic
  - Experiments tracking

* Level 2 (production oriented pipeline):
  - Database usage with data versioning
  - Model versioning
  - Shadow / A/B testing
  - Full CI/CD pipeline
  - Model and data monitoring (infernce time, data/concept drift) - with retraining mechanism (notifications not automatic)
  - Features stores

* Level 3 (Enterprise-grade, scalable, reproducible AI systems):
  - Kubernetes deployment
  - Ray and Anyscale integration
  - Horizontal and vertical scaling (Kubernetes Spark)
  - Canary deployment and rollback strategies
BONUS: custom created real-time monitoring dashboard

# End-to-End ML pipeline/AI System/ with MLE and MLOps Principles (Food-101)
⚠️ This project is a work in progress. Tutorials and features are under active development. Stay tuned for updates and join me on my journey through Machine Learning Engineering!

This projects provides full ML pipeline (with MLOps principles) for food computer vision food classification, implemented to website. For tutorial purpose we will use two methods of deployment FastAPI and CI/CD to handle copying artifacts (onnx model), build docker image and run tests.

# Project requirements and installation guide:
- linux (tested on Ubuntu 22.04)
- docker
https://docs.docker.com/engine/install/ubuntu

- Nvidia GPU/s
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
We will download repo key, configure it (without experimental), update apt, install nvidia-container-toolkit and configure
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update

sudo apt install nvidia-container-toolkit

sudo nvidia-ctk runtime configure --runtime=docker
sudo nvidia-ctk runtime configure --runtime=containerd

sudo systemctl restart docker
```
Now test it - it should display nvidia-smi and check response time for 10 iterations, flops per interaction etc. (benchmark)
```bash
docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi

docker run --rm -it --gpus=all nvcr.io/nvidia/k8s/cuda-sample:nbody nbody -gpu -benchmark
```

- conda or any other venv (remember to switch off poetry default environment creation) for example miniconda:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ~/Miniconda3-latest-Linux-x86_64.sh
```
- poetry (for testing outside docker)
```bash
pipx install poetry
poetry config virtualenvs.create false
```
Check if ~/.config/pypoetry/config.toml was created and have this content:
[virtualenvs]
create = false


Project structure:
 
```
/Food101_MLOps/ 
├── Data/                        # Raw, interim, and processed datasets
│   ├── raw/                     # Original data
│   ├── processed/                # Preprocessed datasets
│   ├── interim/                  # Temporary datasets (optional)
│
├── notebooks/                    # Jupyter notebooks for experimentation, data exploration and also preprocessing. It is also a good place for feature engineering/selection and saving to files, so it could be later automated in e.g. cross-validation in python (not notebook) file.
│
├── src/                          # ML source code
│   ├── __init__.py
│   ├── data/                     # Data loading, preprocessing, augmentation
│   ├── models/                   # Model architectures (multiple CNNs, etc.)
│   ├── training/                 # Training logic, trainers, schedulers
│   ├── evaluation/               # Evaluation metrics plots (like Loss, Accuracy), Metrics computation (Recall, F1 etc.), Testing plots (Confusion matrices, ROC), Generating reports (training and testing) 
│   ├── serving/                  # ONNX export helpers, inference sanity tests
│   ├── utils/                    # Helper functions, common code
│   ├── cli.py                    # CLI entrypoint (train, export, evaluate)
│   ├── config.yaml               # Hyperparameters, architecture selection, paths
│
├── experiments/                   # MLFlow tracking runs
│   ├── mlruns/                    # MLFlow local run storage
│   └── tracking.db                # (optional) Local SQLite DB for MLFlow
│
├── outputs/                        # All model outputs & artifacts
│   ├── checkpoints/                # Exported ONNX or PyTorch models
│   ├── logs/                       # Training logs, console outputs
│   ├── metrics/                    # saved metrics in json format
│   ├── predictions/                # Inference outputs (optional)
│   └── figures/                    # Evaluation plots 
│
├── tests/                          # All tests
│   ├── unit/                       # Unit tests for data, models, training
│   ├── integration/                # Integration tests
│   │   ├── test_exported_model.py  # Validate ONNX inference independently
│   │   ├── test_end_to_end.py      # Run website Docker + send HTTP request
│   │   ├── sample_image.jpg        # Test image for inference
│   │   ├── copy_model.py           # Copy ONNX to website project
│   │   └── start_website.py        # Spin up/down website Docker Compose
│
├── scripts/                        # DevOps & automation scripts
│   ├── deploy_model.sh             # Copy ONNX model to Flask app folder
│   ├── run_local_mlflow.sh         # Start MLFlow UI locally
│   └── entrypoint.sh               # Entrypoint script for Docker modes
│
├── deployment/                      # Deployment & CI/CD infra
│   ├── infrastructure/              # Infra and pipeline configs
│   │   ├── github-actions/          # GitHub Actions YAML workflows
│   │   │   ├── ml_pipeline.yml      # ML training + model deployment
│   │   ├── gitlab-ci/               # (Optional) GitLab CI pipeline configs
│   │   └── docker-compose.yml      # Compose for local orchestration
│   └── docker/                     # Dockerfiles
│       ├── Dockerfile.base         # Base image with CUDA, Python, Poetry
│       ├── Dockerfile.train        # Training container
│       ├── Dockerfile.serve        # FastAPI inference container
│       └── Dockerfile.dev          # Dev notebook environment
│
├── logs/                             # Local dev logs (optional gitignored)
├── inference_api                     # entire logic for program that is using our trained model (FastAPI)
│   ├── main
│   ├── inference
│   ├── config.yaml
│   └── utils/
│       └── postprocessing
│
├── .env                       # Secrets, MLflow URIs, etc.
├── pyproject.toml             # Poetry environment & dependencies
├── MLproject                  # MLflow reproducibility config
├── docker-compose.yml         # High-level orchestration file
└── README.md
```

conda env is needed only for notebooks. It uses the same pyproject.toml that docker containers (later implementation) will be using

### MLOps and ML models life cycle - human in the loop
I see here three main blocks. First is data preparation that could be done in notebooks. User here can: extract/select features, create setup for cross-validation etc. Second is when data is ready, running training and experiments. User here can: choose hyperparameters, change architectures, schedule learning, check training metrics and monitoring artifacts etc. Third, when models are saved is deployment, where the best model is selected converted to onnx and implemented into inference code. The tests are preformed and the implementation of the model and its inference are monitored. This is the end of one life cycle of AI systems, users here monitor and check results of entire pipeline (like Data Scientist, Data Engineers and ML Engineers), but people that edit logic of system on which others work are ML Engineers and if designed points of human interaction with the system are insufficient, ML Engineers modify the project code.

So, to sum up:
* first phase — editing notebooks and processing data. ML Engineer can create Python scripts based on notebooks to automate this part.
* second phase — defining experiments by specifying architectures, editing the config.yaml file, monitoring metrics, and tracking charts.
* third phase — model deployment. Running automation and tracking metrics relevant to the production environment.
Of course, at each of these stages, MLE can make adjustments to the system in accordance with requirements and broadly understood feedback from stakeholders, but the overall structure and interactions with the system remain more or less the same. This production environment can operate continuously. The development process must be consciously guided by humans (even for cyclical retraining, etc.).

This project involves inference of model on website with FastAPI. For simplicity lets host our website and Food101-MLOps project on the same machine. That means we will use docker-compose to orchestrate both website and Food101-MLOps. In real projects, this is not always the case. Sometimes the website and ML pipeline are located on different servers. In such cases, instead of compose, we use Kubernetes, for example.
Docker-compose file will contain only Dockerfile for ML pipeline and Website, but it could also other container like postgres database.

#### How to run:
There are three steps and places controlled by user to interact with project:

In Dockerfile we execute custom scipt which specify MODE of run:
CMD ["bash", "scripts/entrypoint.sh"]

in scripts/entrypoint.sh:
```bash
#!/bin/bash

echo "Running in $MODE mode"

if [ "$MODE" == "train" ]; then
    echo "Starting training..."
    python src/cli.py train --config src/config.yaml

elif [ "$MODE" = "mlflow" ]; then
    echo "Starting MLflow UI..."
    mlflow ui --backend-store-uri experiments/mlruns --host 0.0.0.0 --port 5001

elif [ "$MODE" == "serve" ]; then
    echo "Serving FastAPI model..."
    uvicorn inference_api.main:app --host 0.0.0.0 --port 8000

elif [ "$MODE" == "dev" ]; then
    echo "Starting Jupyter Notebook..."
    jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --NotebookApp.token=''

else
    echo "Invalid mode: $MODE"
    exit 1
fi
```

with the following Dockerfile:
```Dockerfile
# Use Python base image
FROM nvidia/cuda:12.2.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 python3-pip curl git && \
    rm -rf /var/lib/apt/lists/*

# Install poetry
RUN pip install --upgrade pip && pip install poetry

# Disable virtualenvs (important for Docker)
ENV POETRY_VIRTUALENVS_CREATE=false

# Copy poetry project files and install deps
COPY pyproject.toml poetry.lock ./
# install with disabled prompt confirmations and without ANSI colors (clean logs)
RUN poetry install --no-interaction --no-ansi

# Copy entire source code
COPY . .

# Entrypoint
RUN chmod +x scripts/entrypoint.sh
CMD ["bash", "scripts/entrypoint.sh"]
```

```docker-compose:
services:

  train:
    build:
      context: .
      dockerfile: Dockerfile
      args:
        MODE: train
    environment:
      - MODE=train
    runtime: nvidia  # <-- for GPU support
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
    volumes:
      - .:/app
    stdin_open: true
    tty: true

  inference:
    build:
      context: .
      dockerfile: Dockerfile
      args:
        MODE: serve
    environment:
      - MODE=serve
      - MODEL_PATH=/app/outputs/checkpoints/model.onnx
    ports:
      - "8000:8000"
    runtime: nvidia
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
    volumes:
      - .:/app

  notebooks:
    build:
      context: .
      dockerfile: Dockerfile
      args:
        MODE: dev
    environment:
      - MODE=dev
    ports:
      - "8888:8888"
    volumes:
      - .:/app
    runtime: nvidia
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]

  mlflow:
    build:
      context: .
      dockerfile: Dockerfile
      args:
        MODE: mlflow
    environment:
      - MODE=mlflow
    ports:
      - "5001:5001"
    volumes:
      - .:/app
```

1. Data preparation - there are notebooks in which user can prepare data (e.g. feature extraction/selection, cross-validation setup and other data manipulation)
```bash
docker compose up notebooks  # and open website on localhost:8888 to experimient
docker compose stop notebooks
```
2. Running training - when data is ready, user can add architectures, modify config.yaml etc. to customize training and experiments stage.
```bash
docker compose run --rm train  # this command will run train once 
docker compose stop train  # if previous command was executed in background
```

Training outside docker containers
```bash
cd Projekty_py/Food101
PYTHONPATH=. python src/cli.py
mlflow ui --backend-store-uri experiments/mlruns
```

3. Deployment - when experiments are done, the best model is chosen and converted to onnx. Files to edit: 
Two strategies will be presented:
- FastAPI: for simplicity it will be local using other project Blog_CV on local machine
- CI/CD: exporting to Blog_CV but on remote server

```bash
docker compose up -d inference
docker compose stop inference
```

System ML pipeline also supports simultaneous model training and serving, as well as Shadow or A/B testing implementation.

### Adding new architectures
1) src/models - place new files.py with models architectures
2) update config.yaml - architectures


### Data
For educational purposes, we will try to simplify the process as much as possible while retaining the most important concepts of MLOps and MLE. We could, of course, add some embedding model to extract features from images (raw image data is too large) and store them in a database, but this would have additional consequences that would increase the complexity of the project (the use of a database must be justified), e.g.:
- Increasing the complexity of the data flow to provide additional functionality — data/concept drift while maintaining optimal performance

- Introducing dependencies on a persistent storage layer (e.g., a relational or vector database), which requires provisioning, configuration, and integration in both development and production environments.

- Requiring learners to understand embedding models (e.g., CLIP or ResNet), their inference pipelines, and embedding dimensionality — potentially before they've fully grasped the basics of preprocessing and model training.

- Adding the need for vector similarity search, schema versioning, and efficient storage formats — which are advanced concerns that can be explored later.

- Making observability and debugging more difficult, as intermediate representations (like embeddings) are not as intuitively inspectable as raw labels or predictions.

Instead, this tutorial introduces these ideas gradually, and focuses first on getting learners comfortable with the core building blocks: reproducible preprocessing, model training, evaluation, and CI/CD. As learners progress, they’ll be better equipped to appreciate when and why such advanced features like embedding databases are necessary.

Although it could happen for example when user will provide image in negative and then model will have problem classifying. When user provide: 
* images in negative or darker than model is able to detect correctly - we got data drift
* images that are out of scope of known classes - we got concept drift
But that is topic for another lesson - "HOW WE COULD MONITOR DATA DURING INFERNCE?".

Here there is simple data versioning in a form of folders naming. Because of that we will omit data versioning, but for advanced pipelines with databases we should implement it. Because there is no data versioning model versioning doesn't make sense either. Because there are no models to choose from shadow or A/B testing is also pointless. All these simplifications are aimed at make the pipeline easier to understand, so that the operation diagram is not complicated with many branches and loops.

### Inference
To check how it look like localy you can download Test Website repo that uses flask and nginx. For inference OpenCV is used because it got good support for ONNX models (they use optimized runtimes), because inference efficiency in many cases is crucial 

### End
Once you understand how the code works and why it was designed this way, you can move on to analyzing my other projects, where the process is more practical and complete (it creates the life cycle of a working artificial intelligence system).