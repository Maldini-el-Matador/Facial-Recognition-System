"""
Face Recognition Model Training Script
Trains a ResNet18-based face embedding model on the LFW dataset.

Usage:
    python train_model.py

The trained model will be saved to models/face_recognition_model.pth
"""

import os
import json
import time
import numpy as np
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, models
from PIL import Image
from sklearn.datasets import fetch_lfw_people


# ======================== CONFIG ========================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
EMBEDDING_DIM = 128
MIN_IMAGES_PER_PERSON = 20  # Only use identities with >= this many images
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "face_recognition_model.pth")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")


# ======================== MODEL ========================

class FaceEmbeddingNet(nn.Module):
    """
    ResNet18 backbone with a 128-dim embedding head.
    The embedding layer learns discriminative face representations,
    while the classification head provides the training signal.
    """
    def __init__(self, num_classes: int, embedding_dim: int = 128):
        super().__init__()
        
        # Load pretrained ResNet18
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Remove the original FC layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Embedding head: 512 (ResNet18 output) -> 128 (embedding)
        self.embedding = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )
        
        # Classification head for training
        self.classifier = nn.Linear(embedding_dim, num_classes)
    
    def get_embedding(self, x):
        """Extract face embedding (used during inference)."""
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        embedding = self.embedding(features)
        # L2 normalize embeddings
        embedding = nn.functional.normalize(embedding, p=2, dim=1)
        return embedding
    
    def forward(self, x):
        """Forward pass with classification output (used during training)."""
        embedding = self.get_embedding(x)
        logits = self.classifier(embedding)
        return logits, embedding


# ======================== DATA ========================

class LFWFaceDataset(Dataset):
    """
    Custom dataset wrapper for sklearn's LFW data.
    Converts grayscale/numpy images to PIL and applies transforms.
    """
    def __init__(self, images, targets, transform=None):
        self.images = images  # numpy array (N, H, W) or (N, H, W, C)
        self.targets = targets
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        target = self.targets[idx]
        
        # Convert to PIL Image
        if image.ndim == 2:
            # Grayscale - convert to RGB by stacking
            pil_image = Image.fromarray(image.astype(np.uint8), mode='L').convert('RGB')
        elif image.dtype == np.float64 or image.dtype == np.float32:
            # Float images from sklearn - scale to 0-255
            image_uint8 = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
            pil_image = Image.fromarray(image_uint8).convert('RGB')
        else:
            pil_image = Image.fromarray(image).convert('RGB')
        
        if self.transform:
            pil_image = self.transform(pil_image)
        
        return pil_image, target


def get_transforms():
    """Data augmentation and preprocessing transforms."""
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return train_transform, val_transform


def load_lfw_dataset(min_images: int = MIN_IMAGES_PER_PERSON):
    """
    Download and prepare LFW dataset using sklearn.
    Filters to identities with at least `min_images` photos.
    """
    print(f"[*] Downloading/loading LFW dataset (min {min_images} images per person)...")
    print("[*] This may take a few minutes on first run...")
    
    # Use sklearn to fetch LFW - it handles download automatically
    lfw = fetch_lfw_people(
        min_faces_per_person=min_images,
        resize=0.5,  # Half size for faster processing
        color=True,   # Get color images
    )
    
    images = lfw.images  # (N, H, W, C) float64 array, values in [0, 255]
    targets = lfw.target  # integer labels
    target_names = list(lfw.target_names)
    
    num_classes = len(target_names)
    num_images = len(images)
    
    print(f"[+] Dataset loaded: {num_images} images, {num_classes} identities")
    for i, name in enumerate(target_names):
        count = np.sum(targets == i)
        print(f"    {name}: {count} images")
    
    # Get transforms
    train_transform, val_transform = get_transforms()
    
    # Split into train/val (80/20) — stratified
    np.random.seed(42)
    indices = np.arange(num_images)
    np.random.shuffle(indices)
    split = int(0.8 * num_images)
    
    train_indices = indices[:split]
    val_indices = indices[split:]
    
    train_dataset = LFWFaceDataset(images[train_indices], targets[train_indices], transform=train_transform)
    val_dataset = LFWFaceDataset(images[val_indices], targets[val_indices], transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"[+] Train: {len(train_dataset)} images | Val: {len(val_dataset)} images")
    
    return train_loader, val_loader, num_classes, target_names


# ======================== TRAINING ========================

def train_one_epoch(model, train_loader, criterion, optimizer, epoch, num_epochs):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        
        # Forward pass
        logits, embeddings = model(images)
        loss = criterion(logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Stats
        running_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                  f"Loss: {loss.item():.4f} Acc: {100.*correct/total:.1f}%")
    
    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def validate(model, val_loader, criterion):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            logits, _ = model(images)
            loss = criterion(logits, labels)
            
            running_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = running_loss / len(val_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def train_model():
    """Main training function."""
    print("=" * 60)
    print("  Face Recognition Model Training")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Embedding Dim: {EMBEDDING_DIM}")
    print()
    
    # Load data
    train_loader, val_loader, num_classes, target_names = load_lfw_dataset()
    
    # Create model
    model = FaceEmbeddingNet(num_classes=num_classes, embedding_dim=EMBEDDING_DIM).to(DEVICE)
    print(f"\n[+] Model created: ResNet18 + {EMBEDDING_DIM}D embedding + {num_classes} classes")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    
    # Training loop
    best_val_acc = 0.0
    print(f"\n[*] Starting training...\n")
    start_time = time.time()
    
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch, NUM_EPOCHS)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion)
        
        # Step scheduler
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        print(f"\n  Epoch [{epoch+1}/{NUM_EPOCHS}] ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}%")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.1f}%")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
        print()
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(MODEL_DIR, exist_ok=True)
            
            # Save model state dict (embedding part only for inference)
            save_dict = {
                "backbone_state_dict": model.backbone.state_dict(),
                "embedding_state_dict": model.embedding.state_dict(),
                "num_classes": num_classes,
                "embedding_dim": EMBEDDING_DIM,
                "best_val_acc": best_val_acc,
                "epoch": epoch + 1,
            }
            torch.save(save_dict, MODEL_PATH)
            print(f"  [SAVED] Best model saved! (Val Acc: {best_val_acc:.1f}%)")
    
    total_time = time.time() - start_time
    
    # Save metadata
    metadata = {
        "model_type": "ResNet18_FaceEmbedding",
        "embedding_dim": EMBEDDING_DIM,
        "num_classes": num_classes,
        "best_val_accuracy": round(best_val_acc, 2),
        "num_epochs_trained": NUM_EPOCHS,
        "dataset": "LFW (Labeled Faces in the Wild)",
        "min_images_per_person": MIN_IMAGES_PER_PERSON,
        "target_names": target_names,
        "training_time_seconds": round(total_time, 1),
        "device": str(DEVICE),
    }
    
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("=" * 60)
    print(f"  Training Complete!")
    print(f"  Total Time: {total_time/60:.1f} minutes")
    print(f"  Best Val Accuracy: {best_val_acc:.1f}%")
    print(f"  Model saved to: {MODEL_PATH}")
    print(f"  Metadata saved to: {METADATA_PATH}")
    print("=" * 60)
    
    return model, best_val_acc


if __name__ == "__main__":
    train_model()
