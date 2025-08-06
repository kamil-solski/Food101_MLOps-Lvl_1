import matplotlib
matplotlib.use("Agg")  # prevent tkinter-related issues from saving plot - Agg is non-interactive backend. Doesn't require GUI. This headless environemnts are recomended for docker containers, environments wihtout display etc.
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import math

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
    
    return fig

# TODO: should I seprate logic for inference and plot generation?
def plot_roc_ovr(model_dict, test_dataloader, class_names):
    """
    Creates one figure with subplots for each class (OvR ROC), comparing all models.
    Returns: fig, auc_scores dict {class_name: {model_name: auc}}.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Gather predictions from all models
    model_probs = {}  # model_name -> probs (N x C)
    all_labels = []

    for model_name, model in model_dict.items():
        model = model.to(device)
        model.eval()
        all_probs = []
        all_labels = []

        with torch.inference_mode():
            for X_batch, y_batch in test_dataloader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model(X_batch)
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs.cpu())
                all_labels.append(y_batch.cpu())

        model_probs[model_name] = torch.cat(all_probs).numpy()
        all_labels = torch.cat(all_labels).numpy()

    y_bin = label_binarize(all_labels, classes=list(range(len(class_names))))

    # Create subplots grid
    num_classes = len(class_names)
    cols = 3
    rows = math.ceil(num_classes / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = axes.flatten()

    auc_scores = {}

    for idx, class_name in enumerate(class_names):
        ax = axes[idx]
        auc_scores[class_name] = {}

        for model_name, probs in model_probs.items():
            fpr, tpr, _ = roc_curve(y_bin[:, idx], probs[:, idx])
            roc_auc = auc(fpr, tpr)
            auc_scores[class_name][model_name] = roc_auc

            ax.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.3f})")

        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_title(f"ROC - {class_name}")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.grid(True)
        ax.legend()

    # Remove unused axes if any
    for i in range(len(class_names), len(axes)):
        fig.delaxes(axes[i])

    fig.suptitle("ROC Curves Per Class (OvR)", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig, auc_scores


def plot_roc_ovo(model_dict, test_dataloader, class_names, save_path="outputs/figures/roc_overlay_ovo.png"):
    pass