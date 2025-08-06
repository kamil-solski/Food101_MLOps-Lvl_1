import yaml
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path

from src.evaluation import loss_acc_plot, plot_roc_ovr
from src.models import architectures
from src.data.dataloader import get_dataloaders
from src.training.train import train_with_mlflow
from src.utils.paths import DATA_DIR, get_paths, MLFLOW_TRACKING_DIR, clean_outputs_dir, FIGURES_DIR
from src.evaluation.crossval_score import (
    get_all_runs,
    filter_by_loss_discrepancy,
    group_by_arch_and_config,
    score_models,
    select_best_configs
)


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
                        fig = loss_acc_plot(results, output_path=paths["LOSS_ACC_PLOT_PATH"])
                        mlflow.log_figure(fig, artifact_file=paths["LOSS_ACC_PLOT_PATH"].name)
                        plt.close(fig)  # to prevent memory leak if too many plots are created. You won't see them during trainig anyway but in memory they will be saved 
    
    # get the best models for each architecture from cross-validation
    print("\n[INFO] Running automatic model selection from cross-validation results...")
    
    all_runs = get_all_runs(experiment_name=experiment_name)  # current experiment
    filtered_runs = filter_by_loss_discrepancy(runs=all_runs, threshold=0.25)
    grouped_runs = group_by_arch_and_config(filtered_runs, val_metric="val_acc")  # only validation accuracy matter. Training accuracy might be misleading when we would use batch normalization or dropout
    scored = score_models(grouped_runs, val_metric="val_acc", acc_weight=0.7, loss_weight=0.3)  # you can modify weights based on how important you think those metrics sould be
    best_config = select_best_configs(scored)
    
    # Test the best models and compare with ROC AUC
    _, _, test_loader, _ = get_dataloaders(
        image_size=config["image_size"],
        batch_size=config["batch_size"],
        fold="fold0"  # consistent fold for testing. We could use any fold we want or dedicated held-out testing set 
    )
    
    model_dict = {}
    
    for _, row in best_config.iterrows():
        arch = row["architecture"]
        config_str = row["config"]
    
        # Parse config
        hu = int(config_str.split("hu=")[1].split(",")[0])
        lr = float(config_str.split("lr=")[1])

        # Rebuild model
        model_class = architectures[arch]
        model = model_class(input_shape=3, hidden_units=hu, output_shape=len(class_names))
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")

        # Load weights
        model_name = f"{arch}_hu{hu}_lr{lr}"
        paths = get_paths(fold="fold0", model_name=model_name)  # consistent fold
        model.load_state_dict(torch.load(paths["MODEL_CHECKPOINT_PATH"], map_location="cpu"))
        model.eval()

        model_dict[arch] = model
     
    fig, auc_scores = plot_roc_ovr(
        model_dict=model_dict,
        test_dataloader=test_loader,
        class_names=class_names
    )

    # Save locally
    save_path = FIGURES_DIR / "roc_per_class.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)

    # Log to MLflow
    with mlflow.start_run(run_name="evaluate_roc_experiment"):
        mlflow.log_figure(fig, artifact_file="roc_per_class.png")
        for cls, model_aucs in auc_scores.items():
            for model_name, auc_val in model_aucs.items():
                mlflow.log_metric(f"AUC_{cls}_{model_name}", auc_val)

    plt.close(fig)  # optional memory cleanup
        
                            
if __name__ == "__main__":
    clean_outputs_dir()
    main()

# run experiments:
# cd Projekty_py/Food101_MLOps-Lvl_1
# PYTHONPATH=. python src/cli.py
# mlflow ui --backend-store-uri experiments/mlruns

# TODO: fix paths for plots (unfied way of getting paths for plots and model checkpoints). Move saving plots logic to cli.py
# TODO: choose the best overall model from roc tested models
#       * figure out how to write better pipeline for crossval_score.py to include flexible function for selecting best model from metric (accuracy or auc)