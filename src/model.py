import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectrogramCNN(nn.Module):
    def __init__(self):
        super(SpectrogramCNN, self).__init__()
        
        # Convolutional Blocks
        
        # Block 1: Detects simple edges/textures (like the 16kHz cut-off line)
        # Input: (1, 128, 128) becomes Output: (32, 64, 64) after pooling
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1) #the padding here is so that our kernel doesn't "fall off" the image
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) 
        
        # Block 2: Detects shapes such as rectangular artefacts
        # Input: (32, 64, 64) becomes Output: (64, 32, 32) after pooling
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # Block 3: Detects complex concepts such as texture that doesn't seem real
        # Input: (64, 32, 32) becomes Output: (128, 16, 16) after pooling
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        # The Classifier
        
        # Flattening: 128 channels * 16 * 16 pixels = 32,768 features
        self.fc1 = nn.Linear(128 * 16 * 16, 128) # Dense Layer
        self.fc2 = nn.Linear(128, 2)             # Output Layer (2 classes: Bonafide, Spoof)
        
        # Dropout: Randomly turns off 50% of neurons during training to prevent memorization of which ones are real and fake
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Pass through Conv Block 1
        x = self.pool(F.relu(self.conv1(x)))
        
        # Pass through Conv Block 2
        x = self.pool(F.relu(self.conv2(x)))
        
        # Pass through Conv Block 3
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten, 128*16*16 = 32768
        x = torch.flatten(x, 1) 
        
        # Pass through Classifier
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# Simple test to verify architecture dimensions
if __name__ == "__main__":
    # Create a dummy image: (Batch Size 1, 1 Channel, 128 Height, 128 Width)
    testInput = torch.randn(1, 1, 128, 128)
    model = SpectrogramCNN()
    
    # Forward pass
    output = model(testInput)
    
    print(f"Model Architecture Validated.")
    print(f"Input Shape: {testInput.shape}")
    print(f"Output Shape: {output.shape} (Should be [1, 2])")