import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM) implementation.
    Consists of Channel Attention Module (CAM) and Spatial Attention Module (SAM).
    """
    def __init__(self, channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        
        # Channel Attention Module (CAM)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction_ratio, kernel_size=1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(channels // reduction_ratio, channels, kernel_size=1)
        
        # Spatial Attention Module (SAM)
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # Channel Attention
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        channel_att = torch.sigmoid(avg_out + max_out)
        x = x * channel_att # Apply channel attention

        # Spatial Attention
        avg_out_spatial = torch.mean(x, dim=1, keepdim=True)
        max_out_spatial, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.sigmoid_spatial(self.conv_spatial(torch.cat([avg_out_spatial, max_out_spatial], dim=1)))
        
        return x * spatial_att # Apply spatial attention

class MobileNetV2_Attention(nn.Module):
    """
    MobileNetV2 pre-trained model with integrated CBAM attention modules
    and a custom classifier head.
    """
    def __init__(self, num_classes=4):
        super(MobileNetV2_Attention, self).__init__()
        # Load pre-trained MobileNetV2 weights
        self.mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        
        # Integrate CBAM into MobileNetV2's features (convolutional layers)
        # We iterate through the feature layers and insert CBAM after Conv2d layers.
        # This approach replaces the existing module with a Sequential of (original_module, CBAM)
        features_list = []
        for name, module in self.mobilenet.features.named_children():
            features_list.append(module)
            # Apply CBAM after each Conv2d layer
            if isinstance(module, nn.Conv2d):
                channels = module.out_channels
                features_list.append(CBAM(channels))
        self.mobilenet.features = nn.Sequential(*features_list)
        
        # Customize the classifier head for your specific number of classes
        # The last layer of MobileNetV2's original classifier is a Linear layer (1280 -> 1000).
        # We replace the entire classifier with a new Sequential block.
        self.mobilenet.classifier = nn.Sequential(
            nn.Dropout(0.3), # Dropout for regularization
            nn.Linear(1280, num_classes) # Final classification layer
        )
    
    def forward(self, x):
        return self.mobilenet(x)

if __name__ == '__main__':
    # Example usage (for testing this module independently)
    model = MobileNetV2_Attention(num_classes=4)
    print(model)

    # Test with a dummy input
    dummy_input = torch.randn(1, 3, 224, 224) # Batch size 1, 3 channels, 224x224 image
    output = model(dummy_input)
    print(f"Output shape: {output.shape}") # Should be [1, num_classes]