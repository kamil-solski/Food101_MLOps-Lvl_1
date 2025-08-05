import matplotlib
matplotlib.use("Agg")  # prevent tkinter-related issues from saving plot - Agg is non-interactive backend. Doesn't require GUI. This headless environemnts are recomended for docker containers, environments wihtout display etc.
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from sklearn.metrics import roc_curve, auc


def loss_acc_plot(results, output_path):
    epochs = range(len(results['train_loss']))
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    ax[0].plot(epochs, results['train_loss'], label='Train')
    ax[0].plot(epochs, results['val_loss'], label='Validation')
    ax[0].set_title("Loss")
    ax[0].legend()

    ax[1].plot(epochs, results['train_acc'], label='Train')
    ax[1].plot(epochs, results['val_acc'], label='Validation')
    ax[1].set_title("Accuracy")
    ax[1].legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)    
    fig.savefig(output_path)
    
    return fig, str(output_path)

# TODO: make sure that the test data is loaded correctly and that it actually takes the correct loader per fold. In cli.py shouldn't we use once loaded all loaders from get_loaders?
def plot_all_rocs(model_dict, X_test, y_test, save_path="outputs/figures/combined_roc.png"):
    """
    Plots ROC curves for all models and saves to disk. Returns AUC dict and save path.
    """
    plt.figure(figsize=(8, 6))
    aucs = {}

    for name, model in model_dict.items():
        model.eval()
        with torch.no_grad():
            probs = model(X_test)

            # Handle binary classification output
            if probs.shape[1] == 1:
                probs = torch.sigmoid(probs).squeeze(1)
            else:
                probs = torch.softmax(probs, dim=1)[:, 1]  # Binary class prob

            probs = probs.cpu().numpy()
            y_true = y_test.cpu().numpy()

        fpr, tpr, _ = roc_curve(y_true, probs)
        auc_value = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_value:.3f})')
        aucs[name] = auc_value

    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for All Models")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()

    return aucs, str(save_path)