import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit

def get_transforms():
    """Defines the image transformations for training and evaluation."""
    return transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def load_and_split_data(data_dir, test_size=0.2, val_size=0.2, random_state=42):
    """
    Loads the image dataset and splits it into training, validation, and test sets
    using stratified shuffling.

    Args:
        data_dir (str): Path to the root directory of the dataset.
        test_size (float): Proportion of the dataset to include in the test split.
        val_size (float): Proportion of the training/validation set to include in the validation split.
        random_state (int): Seed for random number generation for reproducibility.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset, dataset_classes)
    """
    transform = get_transforms()
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    labels = np.array([label for _, label in dataset.samples])

    # First split: train_val and test
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(sss.split(np.zeros(len(labels)), labels))

    # Second split: train and validation from train_val set
    labels_train_val = labels[train_val_idx]
    sss_train_val = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    train_idx_in_tv, val_idx_in_tv = next(sss_train_val.split(np.zeros(len(labels_train_val)), labels_train_val))

    # Map back to original dataset indices
    train_dataset = Subset(dataset, train_val_idx[train_idx_in_tv])
    val_dataset = Subset(dataset, train_val_idx[val_idx_in_tv])
    test_dataset = Subset(dataset, test_idx)

    return train_dataset, val_dataset, test_dataset, dataset.classes

def get_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=32, num_workers=2):
    """
    Creates PyTorch DataLoader instances for the provided datasets.

    Args:
        train_dataset (torch.utils.data.Subset): Training dataset.
        val_dataset (torch.utils.data.Subset): Validation dataset.
        test_dataset (torch.utils.data.Subset): Test dataset.
        batch_size (int): Batch size for the data loaders.
        num_workers (int): Number of subprocesses to use for data loading.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    # Example usage (for testing this module independently)
    DATA_DIR = '/kaggle/input/augmented-alzheimer-mri-dataset/OriginalDataset' # Adjust for local path
    train_ds, val_ds, test_ds, classes = load_and_split_data(DATA_DIR)
    train_dl, val_dl, test_dl = get_data_loaders(train_ds, val_ds, test_ds)

    print(f"Train dataset size: {len(train_ds)}")
    print(f"Validation dataset size: {len(val_ds)}")
    print(f"Test dataset size: {len(test_ds)}")
    print(f"Classes: {classes}")
