import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import random
from collections import defaultdict
import lime
import lime.lime_image
from skimage.segmentation import mark_boundaries

def plot_sample_images(dataset, num_images_per_class=4, fig_size=(12, 12)):
    """
    Plots a grid of sample images from each class in the dataset.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to sample from.
        num_images_per_class (int): Number of images to plot per class.
        fig_size (tuple): Size of the matplotlib figure.
    """
    classwise_samples = defaultdict(list)
    for idx in range(len(dataset)):
        # Temporarily disable transforms to get original image if needed, or adjust
        # For this plot, we assume `dataset` returns a tensor with applied transforms
        image, label = dataset[idx] 
        classwise_samples[label].append(image)

    num_classes = len(dataset.classes)
    selected_images = []
    selected_labels = []

    for class_idx in range(num_classes):
        images = classwise_samples[class_idx]
        selected_images.extend(random.sample(images, min(num_images_per_class, len(images))))
        selected_labels.extend([class_idx] * min(num_images_per_class, len(images)))

    fig, axes = plt.subplots(num_classes, num_images_per_class, figsize=fig_size)
    axes = axes.flatten()

    # Denormalize and display
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for idx, (image, label) in enumerate(zip(selected_images, selected_labels)):
        image_np = image.permute(1, 2, 0).numpy()
        image_np = image_np * std + mean # Denormalize
        image_np = np.clip(image_np, 0, 1) # Clip to [0, 1] range

        axes[idx].imshow(image_np)
        axes[idx].axis("off")
        axes[idx].set_title(f"Class: {dataset.classes[label]}")

    plt.tight_layout()
    plt.suptitle("Sample Images from Each Class", y=1.02, fontsize=16)
    plt.show()


def plot_class_distribution(train_labels, val_labels, test_labels, class_names):
    """
    Plots the distribution of samples across classes for train, validation, and test sets.

    Args:
        train_labels (list): List of labels in the training set.
        val_labels (list): List of labels in the validation set.
        test_labels (list): List of labels in the test set.
        class_names (list): List of class names.
    """
    from collections import Counter

    train_counts = Counter(train_labels)
    val_counts = Counter(val_labels)
    test_counts = Counter(test_labels)

    x = np.arange(len(class_names))

    plt.figure(figsize=(14, 7))

    bar_width = 0.2
    train_bars = plt.bar(x - bar_width, [train_counts[i] for i in range(len(class_names))], 
                         width=bar_width, label="Train", color="royalblue", alpha=0.8)
    val_bars = plt.bar(x, [val_counts[i] for i in range(len(class_names))], 
                        width=bar_width, label="Validation", color="orangered", alpha=0.8)
    test_bars = plt.bar(x + bar_width, [test_counts[i] for i in range(len(class_names))], 
                        width=bar_width, label="Test", color="seagreen", alpha=0.8)

    # Add counts on top of bars
    for bars in [train_bars, val_bars, test_bars]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height, str(int(height)), 
                     ha='center', va='bottom', fontsize=9, color="black")

    plt.xticks(ticks=x, labels=class_names, rotation=45, ha="right", fontsize=10)
    plt.ylabel("Number of Samples", fontsize=12)
    plt.title("Class Distribution in Train, Validation & Test Sets", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

class GradCAM:
    """
    Implements Grad-CAM for visualizing activation maps.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor, target_class):
        self.model.eval()
        output = self.model(input_tensor)
        self.model.zero_grad()

        # Backpropagate only through the target class
        output[:, target_class].backward()

        gradients = self.gradients.detach().cpu().numpy()
        activations = self.activations.detach().cpu().numpy()
        
        # Global average pooling of gradients to get weights
        weights = np.mean(gradients, axis=(2, 3))[0] # [channels]
        
        # Weighted sum of activations
        cam = np.sum(weights[:, None, None] * activations[0], axis=0) # [H, W]

        # Apply ReLU to remove negative values (only positive contributions)
        cam = np.maximum(cam, 0)
        
        # Normalize to 0-1
        if cam.max() == 0: # Avoid division by zero for completely flat heatmaps
            cam = np.zeros_like(cam)
        else:
            cam -= cam.min()
            cam /= cam.max()

        # Resize heatmap to input image size
        return cv2.resize(cam, (input_tensor.shape[-1], input_tensor.shape[-2]))

def plot_grad_cam(model, test_loader, class_names, target_layer, num_images=16, device='cpu'):
    """
    Plots images with Grad-CAM heatmaps overlaid.

    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for test images.
        class_names (list): List of class names.
        target_layer (nn.Module): The layer to extract activations/gradients from.
        num_images (int): Number of images to display.
        device (str): Device for model inference.
    """
    grad_cam = GradCAM(model, target_layer)
    
    test_images, test_labels = next(iter(test_loader))
    test_images = test_images[:num_images].to(device)
    test_labels = test_labels[:num_images].to(device)

    fig, axes = plt.subplots(int(np.sqrt(num_images)), int(np.sqrt(num_images)), figsize=(12, 12))
    axes = axes.flatten()

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    print("Generating Grad-CAM visualizations...")
    for idx, image in enumerate(test_images):
        input_tensor = image.unsqueeze(0) # Add batch dimension
        
        # Get model prediction for target class
        with torch.inference_mode():
            outputs = model(input_tensor)
            predicted_class = torch.argmax(outputs, dim=1).item()
        
        heatmap = grad_cam.generate_heatmap(input_tensor, predicted_class)
        
        # Denormalize image for display
        image_np = image.permute(1, 2, 0).cpu().numpy()
        image_np = ((image_np * std) + mean) * 255
        image_np = np.clip(image_np, 0, 255).astype(np.uint8)
        
        # Apply colormap to heatmap and overlay
        heatmap_overlay = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Resize heatmap_overlay to match image_np dimensions
        heatmap_overlay = cv2.resize(heatmap_overlay, (image_np.shape[1], image_np.shape[0]))
        
        overlayed_image = cv2.addWeighted(image_np, 0.6, heatmap_overlay, 0.4, 0)
        
        axes[idx].imshow(overlayed_image)
        axes[idx].axis("off")
        axes[idx].set_title(f"True: {class_names[test_labels[idx].item()]}\nPred: {class_names[predicted_class]}")

    plt.tight_layout()
    plt.suptitle("Grad-CAM Visualizations", y=1.02, fontsize=16)
    plt.show()

def plot_lime_explanations(model, test_loader, class_names, num_images=16, device='cpu', num_samples=1000):
    """
    Plots LIME explanations for sample images.

    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for test images.
        class_names (list): List of class names.
        num_images (int): Number of images to display.
        device (str): Device for model inference.
        num_samples (int): Number of samples for LIME explanation.
    """
    explainer = lime.lime_image.LimeImageExplainer()

    def predict_fn(images_np):
        """Wrapper function for LIME: Takes NumPy array, returns probabilities."""
        images_torch = torch.tensor(images_np).permute(0, 3, 1, 2).float().to(device)
        model.eval() # Ensure model is in eval mode during LIME inference
        with torch.inference_mode():
            outputs = model(images_torch)
        return torch.nn.functional.softmax(outputs, dim=1).cpu().detach().numpy()

    test_images, test_labels = next(iter(test_loader))
    test_images = test_images[:num_images]
    test_labels = test_labels[:num_images]

    fig, axes = plt.subplots(int(np.sqrt(num_images)), int(np.sqrt(num_images)), figsize=(12, 12))
    axes = axes.flatten()

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    print("Generating LIME explanations (this may take a while)...")
    for idx, image in enumerate(test_images):
        image_np_denorm = image.permute(1, 2, 0).cpu().numpy()
        image_np_denorm = (image_np_denorm * std) + mean # Denormalize for LIME

        # Get model prediction for the original image
        with torch.inference_mode():
            original_image_tensor = image.unsqueeze(0).to(device)
            predicted_class = torch.argmax(model(original_image_tensor), dim=1).item()

        # Generate LIME explanation for the predicted class
        explanation = explainer.explain_instance(
            image_np_denorm, 
            predict_fn, 
            labels=(predicted_class,), # Explain for the predicted class
            top_labels=None, # Only interested in the predicted class
            hide_color=0, 
            num_samples=num_samples
        )
        
        # Get image and mask for the predicted class
        temp, mask = explanation.get_image_and_mask(
            predicted_class, 
            positive_only=False, 
            num_features=5, # Show top 5 important features
            hide_rest=True # Hide non-relevant parts
        )
        
        axes[idx].imshow(mark_boundaries(temp / temp.max(), mask)) # Normalize temp for display
        axes[idx].axis("off")
        axes[idx].set_title(f"True: {class_names[test_labels[idx].item()]}\nPred: {class_names[predicted_class]}")

    plt.tight_layout()
    plt.suptitle("LIME Explanations", y=1.02, fontsize=16)
    plt.show()