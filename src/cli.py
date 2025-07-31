import yaml
from src.models import architectures
from src.data.dataloader import get_dataloaders
from src.training.train import train_with_mlflow
from src.utils.paths import DATA_DIR, get_paths
import torch
import torch.nn as nn
import torch.optim as optim

def main():
    with open("src/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    dataset_name = config["dataset"]
    dataset_path = DATA_DIR / dataset_name

    # List folds dynamically
    folds = sorted([d for d in dataset_path.iterdir() if d.is_dir() and d.name.startswith("fold")])

    for fold in folds:
        print(f"\n[INFO] Running Fold: {fold}")

        train_loader, val_loader, class_names = get_dataloaders(
            image_size=config["image_size"],
            batch_size=config["batch_size"],
            fold=fold
        )

        for arch_name in config["architectures"]:
            model_class = architectures[arch_name]
            for hu in config["hidden_units"]:
                for lr in config["learning_rates"]:
                    model = model_class(input_shape=3, hidden_units=hu, output_shape=len(class_names))
                    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

                    model_name = f"{arch_name}_hu{hu}_lr{lr}"  # we want only architecture name here because full path will break traing loop
                    
                    paths = get_paths(fold=fold.name, model_name=model_name)
 
                    
                    train_with_mlflow(
                        model=model,
                        model_name=model_name,
                        train_dataloader=train_loader,
                        val_dataloader=val_loader,
                        loss_fn=nn.CrossEntropyLoss(),
                        optimizer=optim.Adam(model.parameters(), lr=lr),
                        learning_rate=lr,
                        epochs=config["num_epochs"],
                        paths=paths,
                        run_name=None
                    )

if __name__ == "__main__":
    main()

# run experiments:
# cd Projekty_py/Food101
# PYTHONPATH=. python src/cli.py
# mlflow ui --backend-store-uri experiments/mlruns