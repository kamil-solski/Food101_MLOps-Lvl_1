"""
Tags are essential in our use case (especially architecture and fold) bacasuse they allow us to filter neccessary runs to extract the best hyperparameter combination from folds.
Because we got folds we need to get averages per hyperparameter combination across folds (for each architecture).
So, from num_architectures * num_folds * num_combination we should get num_architectures of models.
We will filter by tags, averege per combination and pick the highest value for each architecture.
"""
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

# 
def get_all_runs(experiment_name: str) -> pd.DataFrame:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")
    
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        output_format="pandas"
    )
    return runs

# 
def filter_by_loss_discrepancy(runs: pd.DataFrame, threshold: float = 0.25) -> pd.DataFrame:
    runs["loss_diff"] = (runs["metrics.train_loss"] - runs["metrics.val_loss"]).abs()
    return runs[runs["loss_diff"] <= threshold].copy()

# 
def group_by_arch_and_config(runs: pd.DataFrame, val_metric: str = "val_acc") -> pd.DataFrame:
    grouped = runs.groupby(["tags.architecture", "tags.config"])

    summary = []
    for (arch, config), group in grouped:
        avg_val_loss = group["metrics.val_loss"].mean()
        avg_val_metric = group[f"metrics.{val_metric}"].mean()

        summary.append({
            "architecture": arch,
            "config": config,
            "folds": len(group),
            "avg_val_loss": avg_val_loss,
            f"avg_{val_metric}": avg_val_metric
        })

    return pd.DataFrame(summary)

# 
def score_models(df: pd.DataFrame, val_metric: str = "val_acc",
                 acc_weight: float = 0.7, loss_weight: float = 0.3) -> pd.DataFrame:
    df = df.copy()

    # Normalize
    df["loss_score"] = 1 - (df["avg_val_loss"] - df["avg_val_loss"].min()) / (df["avg_val_loss"].max() - df["avg_val_loss"].min())
    df["metric_score"] = (df[f"avg_{val_metric}"] - df[f"avg_{val_metric}"].min()) / (df[f"avg_{val_metric}"].max() - df[f"avg_{val_metric}"].min())

    # Weighted score
    df["score"] = acc_weight * df["metric_score"] + loss_weight * df["loss_score"]
    return df.sort_values("score", ascending=False)

#
def select_best_configs(df_scored: pd.DataFrame) -> pd.DataFrame:
    return df_scored.loc[df_scored.groupby("architecture")["score"].idxmax()].reset_index(drop=True)