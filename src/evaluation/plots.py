import matplotlib.pyplot as plt
from pathlib import Path
import mlflow
import pandas as pd


def save_loss_plot(results, model_name, output_dir):
    epochs = range(len(results['train_loss']))
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    ax[0].plot(epochs, results['train_loss'], label='Train')
    ax[0].plot(epochs, results['test_loss'], label='Test')
    ax[0].set_title("Loss")
    ax[0].legend()

    ax[1].plot(epochs, results['train_acc'], label='Train')
    ax[1].plot(epochs, results['test_acc'], label='Test')
    ax[1].set_title("Accuracy")
    ax[1].legend()

    plot_path = output_dir / f"{model_name}_loss_plot.png"
    fig.savefig(plot_path)
    plt.close(fig)  # to prevent memory leak if too many plots are created. You won't see them during trainig anyway but in memory they will be saved 
    return str(plot_path)

def plot_model_comparison(model_results: dict, model_labels: list[str], out_name: str = "comparison_plot.png", output_dir: Path = None):
    plt.figure(figsize=(15, 10))
    epochs = range(len(next(iter(model_results.values()))["train_loss"]))

    for i, label in enumerate(model_labels):
        df = pd.DataFrame(model_results[label])
        plt.subplot(2, 2, 1)
        plt.plot(epochs, df["train_loss"], label=label)
        plt.subplot(2, 2, 2)
        plt.plot(epochs, df["test_loss"], label=label)
        plt.subplot(2, 2, 3)
        plt.plot(epochs, df["train_acc"], label=label)
        plt.subplot(2, 2, 4)
        plt.plot(epochs, df["test_acc"], label=label)

    plt.subplot(2, 2, 1); plt.title("Train Loss"); plt.xlabel("Epochs"); plt.legend()
    plt.subplot(2, 2, 2); plt.title("Test Loss"); plt.xlabel("Epochs"); plt.legend()
    plt.subplot(2, 2, 3); plt.title("Train Accuracy"); plt.xlabel("Epochs"); plt.legend()
    plt.subplot(2, 2, 4); plt.title("Test Accuracy"); plt.xlabel("Epochs"); plt.legend()

    plot_file = output_dir / out_name
    plt.savefig(plot_file)
    plt.close()

    # log to current MLFlow run (if inside one)
    if mlflow.active_run():
        mlflow.log_artifact(str(plot_file), artifact_path="comparison_plots")

    return str(plot_file)

def compare_top_models(metric_name="test_acc", top_n=2, experiment_name="Default", comparison_run_name="Model_Comparison"):
    # Set experiment
    mlflow.set_experiment(experiment_name)

    # Fetch top N runs
    df = mlflow.search_runs(order_by=[f"metrics.{metric_name} DESC"])
    top_runs = df.head(top_n)

    print("Top runs being compared:")
    print(top_runs[["run_name", f"metrics.{metric_name}", "params.model", "params.learning_rate"]])

    # Load run results from logged artifacts
    model_results = {}
    for _, row in top_runs.iterrows():
        run_id = row["run_id"]
        run_name = row["run_name"]
        model_name = f"{row['params.model']}_lr{row['params.learning_rate']}_hu{row['params.hidden_units']}"
        
        # Load the artifact (you must ensure the metrics were saved as a .json or .csv)
        artifact_path = f"mlruns/{row['experiment_id']}/{run_id}/artifacts/results.csv"
        if Path(artifact_path).exists():
            df = pd.read_csv(artifact_path)
            model_results[model_name] = df

    # Now plot and log the comparison
    if model_results:
        with mlflow.start_run(run_name=comparison_run_name):
            fig_path = plot_model_comparison(model_results, list(model_results.keys()))
            mlflow.log_artifact(fig_path, artifact_path="comparisons")