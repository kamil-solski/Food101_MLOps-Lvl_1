import yaml
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import matplotlib.pyplot as plt

from src.evaluation import loss_acc_plot, plot_all_rocs
from src.evaluation.crossval_score import avg_combinations
from src.models import architectures
from src.data.dataloader import get_dataloaders
from src.training.train import train_with_mlflow
from src.utils.paths import DATA_DIR, get_paths, MLFLOW_TRACKING_DIR, clean_outputs_dir

mlflow.set_tracking_uri(MLFLOW_TRACKING_DIR.as_uri())

def main():
    with open("src/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    dataset_name = config["dataset"]
    data_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = f"{dataset_name}_{data_str}"  # with this it should be used strategically (with changing hyperparameters for each experiment)
    
    mlflow.set_experiment(experiment_name)
    
    dataset_path = DATA_DIR / dataset_name
    folds = sorted([d for d in dataset_path.iterdir() if d.is_dir() and d.name.startswith("fold")])

    for fold in folds:
        print(f"\n[INFO] Running: {fold.name}")

        train_loader, val_loader, _, class_names = get_dataloaders(
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
                    run_name = f"{model_name}_{fold.name}"
                    
                    paths = get_paths(fold=fold.name, model_name=model_name)
 
                    with mlflow.start_run(run_name=run_name):
                        mlflow.set_tag("fold", fold.name)
                        mlflow.set_tag("architecture", arch_name)
                        mlflow.set_tag("config", f"hu={hu},lr={lr}")                    
                    
                        results = train_with_mlflow(
                            model=model,
                            model_name=model_name,
                            train_dataloader=train_loader,
                            val_dataloader=val_loader,
                            loss_fn=nn.CrossEntropyLoss(),
                            optimizer=optim.Adam(model.parameters(), lr=lr),
                            learning_rate=lr,
                            epochs=config["num_epochs"],
                            paths=paths
                        )
                        
                        # Save training curve as image
                        fig, loss_acc_plot_path = loss_acc_plot(results, output_path=paths["LOSS_ACC_PLOT_PATH"])
                        mlflow.log_figure(fig, artifact_file=paths["LOSS_ACC_PLOT_PATH"].name)
                        plt.close(fig)  # to prevent memory leak if too many plots are created. You won't see them during trainig anyway but in memory they will be saved 
            
                            
if __name__ == "__main__":
    clean_outputs_dir()
    main()

# run experiments:
# cd Projekty_py/Food101_MLOps-Lvl_1
# PYTHONPATH=. python src/cli.py
# mlflow ui --backend-store-uri experiments/mlruns