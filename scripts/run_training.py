import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os

# Import modules from src/
from src.data_loader import load_and_split_data, get_data_loaders
from src.models import MobileNetV2_Attention
from src.train import train_model
from src.evaluate import plot_metrics, evaluate_model_with_metrics
from src.visualize_xai import plot_sample_images, plot_class_distribution

def main():
    # --- Configuration ---
    DATA_DIR = '/kaggle/input/augmented-alzheimer-mri-dataset/OriginalDataset' # Adjust for your local path
    MODEL_SAVE_PATH = 'trained_models/mobilenetv2_cbam_best.pth'
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    EARLY_STOP_PATIENCE = 10
    NUM_CLASSES = 4 # Adjust based on your dataset classes
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # Ensure save directory exists
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    # --- Data Loading and Splitting ---
    print("Loading and splitting data...")
    train_dataset, val_dataset, test_dataset, class_names = load_and_split_data(
        data_dir=DATA_DIR, test_size=0.2, val_size=0.2, random_state=42
    )
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")
    print(f"Class names: {class_names}")

    train_loader, val_loader, test_loader = get_data_loaders(
        train_dataset, val_dataset, test_dataset, batch_size=BATCH_SIZE, num_workers=2
    )

    # --- Data Visualization (Optional, run once to check splits) ---
    # To get original labels for plotting, access dataset.dataset.targets
    # train_labels_for_plot = [train_dataset.dataset.targets[i] for i in train_dataset.indices]
    # val_labels_for_plot = [val_dataset.dataset.targets[i] for i in val_dataset.indices]
    # test_labels_for_plot = [test_dataset.dataset.targets[i] for i in test_dataset.indices]
    
    # print("\nPlotting sample images...")
    # plot_sample_images(train_dataset.dataset, num_images_per_class=4) # Use the base dataset for plotting

    # print("\nPlotting class distribution...")
    # plot_class_distribution(train_labels_for_plot, val_labels_for_plot, test_labels_for_plot, class_names)


    # --- Model, Loss, Optimizer, Scheduler Setup ---
    print("\nSetting up model, loss, optimizer, and scheduler...")
    model = MobileNetV2_Attention(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    # --- Training ---
    print("\nStarting model training...")
    trained_model, train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, val_loader, criterion, optimizer, lr_scheduler, 
        epochs=NUM_EPOCHS, device=DEVICE, early_stop_patience=EARLY_STOP_PATIENCE,
        save_path=MODEL_SAVE_PATH
    )

    # --- Plot Training Metrics ---
    print("\nPlotting training and validation metrics...")
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)

    # --- Final Evaluation on Test Set ---
    print("\nEvaluating model on the test set...")
    # Load the best model for final evaluation
    best_model = MobileNetV2_Attention(num_classes=NUM_CLASSES).to(DEVICE)
    best_model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    evaluate_model_with_metrics(best_model, test_loader, class_names, device=DEVICE)

    print("\nTraining and evaluation process complete!")

if __name__ == "__main__":
    main()
