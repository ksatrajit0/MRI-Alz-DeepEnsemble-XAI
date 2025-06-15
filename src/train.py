import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import time

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=50,
                device='cpu', early_stop_patience=10, save_path="best_model.pth"):
    """
    Trains the deep learning model.

    Args:
        model (nn.Module): The model to be trained.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer for model parameters.
        scheduler (LRScheduler): Learning rate scheduler.
        epochs (int): Number of training epochs.
        device (str): Device to run training on ('cuda' or 'cpu').
        early_stop_patience (int): Number of epochs to wait for improvement before stopping.
        save_path (str): Path to save the best model checkpoint.

    Returns:
        nn.Module: The trained model.
        tuple: (train_losses, val_losses, train_accuracies, val_accuracies)
    """
    best_val_loss = float("inf")
    early_stop_counter = 0

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        avg_train_loss = running_loss / total # Average loss per sample

        val_acc, avg_val_loss = evaluate_model(model, val_loader, criterion, device)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        scheduler.step(avg_val_loss) # Step scheduler based on validation loss

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs} | Time: {epoch_time:.2f}s | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), save_path) # Save best model
            print(f"Saved best model to {save_path} (Validation Loss: {best_val_loss:.4f})")
        else:
            early_stop_counter += 1
            print(f"Early stopping counter: {early_stop_counter}/{early_stop_patience}")

        if early_stop_counter >= early_stop_patience:
            print("Early stopping triggered! Training halted.")
            break

    return model, train_losses, val_losses, train_accuracies, val_accuracies

def evaluate_model(model, loader, criterion, device='cpu'):
    """
    Evaluates the model on a given DataLoader.

    Args:
        model (nn.Module): The model to evaluate.
        loader (DataLoader): DataLoader for the evaluation set.
        criterion (nn.Module): Loss function.
        device (str): Device to run evaluation on ('cuda' or 'cpu').

    Returns:
        tuple: (accuracy, average_loss)
    """
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0

    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = running_loss / total # Average loss per sample
    accuracy = correct / total
    return accuracy, avg_loss