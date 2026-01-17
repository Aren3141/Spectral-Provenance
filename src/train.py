import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import os
from model import SpectrogramCNN

# --- CONFIGURATION ---
BATCH_SIZE = 32         # How many images we process at once
LEARNING_RATE = 0.001   # How big our steps "down the hill" are
EPOCHS = 10             # How many times we show the entire dataset to the model
DATA_PATH = "data/processed" # Where your images are

def train():
    # 1. SETUP DEVICE
    # We prioritize Apple Silicon (MPS) for your M3 Air, then CUDA, then CPU.
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(">>> Using Apple MPS Acceleration.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(">>> Using CUDA Acceleration.")
    else:
        device = torch.device("cpu")
        print(">>> Using CPU (Warning: Slow).")

    # 2. DATA PREPARATION
    # We convert the images to Tensors (Numbers) AND resize them.
    transform = transforms.Compose([
        transforms.Resize((128, 128)),               #  ADD THIS LINE
        transforms.Grayscale(num_output_channels=1), # Ensure it is 1 channel
        transforms.ToTensor()
    ])

    # ImageFolder automatically labels data based on folder names (bonafide vs spoof)
    print(">>> Loading Dataset...")
    train_dataset = datasets.ImageFolder(root=DATA_PATH, transform=transform)
    
    # The DataLoader shuffles the data and feeds it to the model in batches
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"    Found {len(train_dataset)} images.")
    print(f"    Classes: {train_dataset.classes}") # Should be ['bonafide', 'spoof']

    # 3. INITIALIZE THE BRAIN
    model = SpectrogramCNN().to(device)
    
    # 4. MATH SETUP
    criterion = nn.CrossEntropyLoss()      # The Error Calculator
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) # The Learner

    # 5. THE TRAINING LOOP
    print("\n>>> Starting Training...")
    
    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        start_time = time.time()

        for i, (images, labels) in enumerate(train_loader):
            # Move data to the M3 Chip
            images, labels = images.to(device), labels.to(device)

            # Zero the gradients (reset calculus from previous step)
            optimizer.zero_grad()

            # Forward Pass: Ask the model to guess
            outputs = model(images)

            # Calculate Loss: How wrong was it?
            loss = criterion(outputs, labels)

            # Backward Pass: Calculate calculus gradients (Backpropagation)
            loss.backward()

            # Optimize: Adjust weights
            optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        # End of Epoch Stats
        epoch_acc = (correct_predictions / total_predictions) * 100
        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Loss: {running_loss/len(train_loader):.4f} | "
              f"Accuracy: {epoch_acc:.2f}% | "
              f"Time: {epoch_time:.2f}s")

    # 6. SAVE THE BRAIN
    print("\n>>> Training Complete.")
    if not os.path.exists("models"):
        os.makedirs("models")
    
    torch.save(model.state_dict(), "models/spectral_cnn_v1.pth")
    print(">>> Model saved to models/spectral_cnn_v1.pth")

if __name__ == "__main__":
    train()