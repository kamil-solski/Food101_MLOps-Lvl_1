import torch
import os
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.models import architectures
from src.utils.paths import get_paths
from src.serving.export_to_onnx import export_model_to_onnx

def evaluate_model(model, dataloader):
    model.eval()
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to("cpu")
            y = y.to("cpu")
            probs = torch.softmax(model(x), dim=1)[:, 1]  # assuming binary classification
            all_probs.extend(probs.tolist())
            all_targets.extend(y.tolist())
    return roc_auc_score(all_targets, all_probs)

def main():
    # Load config
    with open("src/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    dataset = config["dataset"]
    image_size = config["image_size"]
    fold_list = sorted(os.listdir(f"Data/{dataset}"))
    best_models = []

    for fold in fold_list:
        paths = get_paths(fold=fold)
        test_dir = paths["TEST_DIR"]
        classes_file = paths["CLASSES_FILE"]

        # Load test data
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])
        test_data = datasets.ImageFolder(test_dir, transform=transform)
        test_loader = DataLoader(test_data, batch_size=32)

        # Load class count
        with open(classes_file) as f:
            class_names = [line.strip() for line in f.readlines()]
        num_classes = len(class_names)

        # Find models in fold checkpoint dir
        fold_ckpt_dir = paths["CHECKPOINTS_DIR"] / dataset / fold
        auc_scores = []

        for filename in os.listdir(fold_ckpt_dir):
            if not filename.endswith(".pt"):
                continue
            model_path = fold_ckpt_dir / filename
            arch = filename.split("_")[0]
            hu = int(filename.split("_hu")[1].split("_")[0])
            lr = float(filename.split("_lr")[1].replace(".pt", ""))
            model = architectures[arch](input_shape=3, hidden_units=hu, output_shape=num_classes)
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            model.eval()

            auc = evaluate_model(model, test_loader)
            auc_scores.append((filename, auc))

        best_model_name, best_auc = sorted(auc_scores, key=lambda x: x[1], reverse=True)[0]
        print(f"[✓] Best model for {fold}: {best_model_name} (AUC={best_auc:.4f})")
        best_models.append((fold, best_model_name, arch, hu, lr))

        # Export best model to ONNX
        export_model_to_onnx(
            model_name=best_model_name.replace(".pt", ""),
            arch_name=arch,
            hidden_units=hu,
            num_classes=num_classes,
            image_size=image_size,
            fold=fold
        )

if __name__ == "__main__":
    main()
