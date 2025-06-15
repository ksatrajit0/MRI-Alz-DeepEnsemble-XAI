import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm

def extract_features(model, data_loader, device='cpu', output_csv_path="features.csv"):
    """
    Extracts deep features from a given model for the entire dataset.

    Args:
        model (nn.Module): The trained model to extract features from.
        data_loader (DataLoader): DataLoader for the entire dataset.
        device (str): Device to run inference on ('cuda' or 'cpu').
        output_csv_path (str): Path to save the extracted features as a CSV.
    """
    # Temporarily modify the model's classifier to output features instead of logits
    original_classifier = model.mobilenet.classifier
    model.mobilenet.classifier = nn.Identity() # Replace classifier with Identity to get features before final classification

    model.to(device)
    model.eval()

    features_list = []
    labels_list = []

    print(f"Extracting features to {output_csv_path}...")
    with torch.inference_mode():
        for images, labels in tqdm(data_loader):
            images = images.to(device)
            feature_vectors = model(images)
            features_list.extend(feature_vectors.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

    # Restore original classifier (important if you want to use the model for inference later)
    model.mobilenet.classifier = original_classifier
    
    features_df = pd.DataFrame(features_list)
    features_df['label'] = labels_list
    features_df.to_csv(output_csv_path, index=False)
    print("Feature extraction complete!")