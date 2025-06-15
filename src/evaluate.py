# src/evaluate.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model_with_metrics(model, loader, class_names, device='cpu'):
    """
    Evaluates the model and prints a classification report and plots a confusion matrix.

    Args:
        model (nn.Module): The trained model to evaluate.
        loader (DataLoader): DataLoader for the evaluation set.
        class_names (list): List of class names for reporting.
        device (str): Device to run evaluation on ('cuda' or 'cpu').
    """
    model.eval()
    model.to(device)
    all_preds = []
    all_labels = []

    print("Generating evaluation metrics...")
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Classification Report
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print("\n--- Classification Report ---")
    print(report)
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix Heatmap")
    plt.show()

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies):
    """
    Plots the training and validation loss and accuracy curves over epochs.

    Args:
        train_losses (list): List of training losses per epoch.
        val_losses (list): List of validation losses per epoch.
        train_accuracies (list): List of training accuracies per epoch.
        val_accuracies (list): List of validation accuracies per epoch.
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(14, 6))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss", marker="o", linestyle='-', color='skyblue')
    plt.plot(epochs, val_losses, label="Validation Loss", marker="o", linestyle='--', color='salmon')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    
    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label="Train Accuracy", marker="o", linestyle='-', color='lightgreen')
    plt.plot(epochs, val_accuracies, label="Validation Accuracy", marker="o", linestyle='--', color='lightcoral')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training & Validation Accuracy")
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()

    plt.tight_layout()
    plt.show()