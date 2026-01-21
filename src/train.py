import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import os
# We import the model structure you already have
from model import SpectrogramCNN 

# --- CONFIGURATION ---
BATCH_SIZE = 64         # Increased to 64 for M3 efficiency (speeds up training)
LEARNING_RATE = 0.001   
EPOCHS = 10             
DATA_PATH = "data/processed" 
MODEL_SAVE_PATH = "models/spectral_cnn_v2.pth" # Saving as V2

def train():
    # 1. SETUP DEVICE
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(">>> [SYSTEM] Apple M3 MPS Acceleration Engaged.")
    else:
        device = torch.device("cpu")
        print(">>> [SYSTEM] Using CPU (Warning: Slower).")

    # 2. DATA PREPARATION
    # transformation must match processor.py output (Grayscale)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),               
        transforms.Grayscale(num_output_channels=1), 
        transforms.ToTensor()
    ])

    print(f">>> [DATA] Loading Dataset from {DATA_PATH}...")
    
    # Check if data exists first
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] {DATA_PATH} does not exist. Did processor.py finish?")
        return

    train_dataset = datasets.ImageFolder(root=DATA_PATH, transform=transform)
    
    # We shuffle to prevent the model from learning the order of files
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f">>> [DATA] Loaded {len(train_dataset)} segments.")
    print(f">>> [DATA] Classes detected: {train_dataset.classes}")

    # 3. INITIALIZE MODEL
    model = SpectrogramCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. TRAINING LOOP
    print(f">>> [TRAINING] Beginning {EPOCHS} Epochs on M3 Neural Engine...")
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        model.train() # Set model to training mode (enables Dropout)

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()           # Reset gradients
            outputs = model(images)         # Forward pass
            loss = criterion(outputs, labels) # Calculate error
            loss.backward()                 # Backward pass (Calculus)
            optimizer.step()                # Update weights

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        # Statistics per Epoch
        epoch_acc = (correct_predictions / total_predictions) * 100
        epoch_time = time.time() - start_time
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Loss: {running_loss/len(train_loader):.4f} | "
              f"Accuracy: {epoch_acc:.2f}% | "
              f"Time: {epoch_time:.2f}s")

    # 5. SAVE
    print("\n>>> [COMPLETE] Training Finished.")
    if not os.path.exists("models"):
        os.makedirs("models")
    
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f">>> [SAVED] Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()