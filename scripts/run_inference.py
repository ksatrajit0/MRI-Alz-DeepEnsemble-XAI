import torch
import os

# Import modules from src/
from src.data_loader import load_and_split_data, get_data_loaders, get_transforms
from src.models import MobileNetV2_Attention
from src.visualiz_xai import plot_grad_cam, plot_lime_explanations
from src.feature_extractor import extract_features
from torchvision import datasets # For full dataset for feature extraction

def main():
    # --- Configuration ---
    DATA_DIR = '/kaggle/input/augmented-alzheimer-mri-dataset/OriginalDataset' # Adjust for your local path
    MODEL_PATH = 'trained_models/mobilenetv2_cbam_best.pth'
    FEATURES_OUTPUT_PATH = 'data/trained_mobilenet_cbam_features.csv'
    NUM_CLASSES = 4
    BATCH_SIZE = 32
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model checkpoint not found at {MODEL_PATH}. Please train the model first.")
        return

    # --- Load Model ---
    print("Loading trained model...")
    model = MobileNetV2_Attention(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("Model loaded successfully!")

    # --- Data Loaders for Inference/Visualization ---
    # For visualization, we often use the test_loader, but for feature extraction
    # we might want features for the entire dataset.
    _, _, test_dataset, class_names = load_and_split_data(
        data_dir=DATA_DIR, test_size=0.2, val_size=0.2, random_state=42
    )
    test_loader = get_data_loaders(None, None, test_dataset, batch_size=BATCH_SIZE)[2] # Get only test_loader

    # --- Grad-CAM Visualization ---
    print("\nGenerating Grad-CAM visualizations...")
    # Identify the target layer for Grad-CAM. For MobileNetV2, features[-3] (invblock_15) or features[-1] (Conv2d before pooling)
    # usually works well. Adjust if needed.
    target_layer_grad_cam = model.mobilenet.features[-3] 
    plot_grad_cam(model, test_loader, class_names, target_layer_grad_cam, num_images=16, device=DEVICE)

    # --- LIME Explanations ---
    print("\nGenerating LIME explanations...")
    plot_lime_explanations(model, test_loader, class_names, num_images=16, device=DEVICE, num_samples=1000)

    # --- Feature Extraction ---
    print("\nStarting feature extraction...")
    # For feature extraction, we want to load the *full* dataset without train/val/test splits
    # to extract features for all samples.
    full_transform = get_transforms() # Use the same transformations
    full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=full_transform)
    full_data_loader = get_data_loaders(full_dataset, None, None, batch_size=BATCH_SIZE, num_workers=2)[0] # Get train_loader as full_data_loader

    # Ensure data directory for features exists
    os.makedirs(os.path.dirname(FEATURES_OUTPUT_PATH), exist_ok=True)
    extract_features(model, full_data_loader, device=DEVICE, output_csv_path=FEATURES_OUTPUT_PATH)

    print("\nInference, visualization, and feature extraction complete!")

if __name__ == "__main__":
    main()