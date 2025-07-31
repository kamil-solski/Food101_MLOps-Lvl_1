from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

from src.utils.paths import get_paths


def get_dataloaders(image_size: int, batch_size: int, fold: str = None):
    paths = get_paths(fold=fold)   # dataloader need to extract datasets from specific folder structure so fold has to be specified
    
    train_dir = paths["TRAIN_DIR"]
    val_dir = paths["VAL_DIR"]
    classes_file = paths["CLASSES_FILE"]

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    # Load datasets
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    val_data = datasets.ImageFolder(val_dir, transform=transform)

    # Load class names manually from file
    with open(classes_file, "r") as f:
        class_names = [line.strip() for line in f.readlines()]

    # Sanity check
    assert class_names == train_data.classes, "Mismatch between loaded class names and dataset labels"

    # Setup DataLoaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count() // 4)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count() // 4)

    return train_loader, val_loader, class_names
