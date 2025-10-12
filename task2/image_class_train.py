"""
Image Classification Model Training Script for Animal Recognition
Uses transfer learning with ResNet50 - Fixed version with all recommendations
"""

import argparse
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter

# =============================================================================
# DATASET
# =============================================================================

class AnimalDataset(Dataset):
    """Custom dataset for animal images"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def load_dataset(data_dir, class_names):
    """Load dataset from directory structure"""
    image_paths = []
    labels = []
    
    class_to_idx = {name: idx for idx, name in enumerate(sorted(class_names))}
    
    print("\nLoading dataset...")
    for class_name in tqdm(os.listdir(data_dir)):
        class_path = os.path.join(data_dir, class_name)
        
        if not os.path.isdir(class_path) or class_name not in class_to_idx:
            continue
        
        label = class_to_idx[class_name]
        
        for img_file in os.listdir(class_path):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(class_path, img_file)
                image_paths.append(img_path)
                labels.append(label)
    
    print(f"Loaded {len(image_paths)} images from {len(class_to_idx)} classes")
    return image_paths, labels, class_to_idx


def calculate_class_weights(labels, num_classes):
    """Calculate class weights for imbalanced dataset"""
    label_counts = Counter(labels)
    total_samples = len(labels)
    
    # Calculate weights: inverse of class frequency
    weights = []
    for i in range(num_classes):
        count = label_counts.get(i, 1)
        weight = total_samples / (num_classes * count)
        weights.append(weight)
    
    return torch.FloatTensor(weights)


# =============================================================================
# MODEL
# =============================================================================

def create_model(num_classes, pretrained=True):
    """Create ResNet50 model for transfer learning"""
    model = models.resnet50(pretrained=pretrained)
    
    # Freeze early layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace classifier with custom head
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    
    return model


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc='Training')
    
    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Validating'):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train_classifier(args):
    print("=" * 80)
    print("ANIMAL IMAGE CLASSIFICATION TRAINING")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    class_names = ['dog', 'cat', 'horse', 'spider', 'butterfly', 'chicken',
                   'sheep', 'cow', 'squirrel', 'elephant']
    image_paths, labels, class_to_idx = load_dataset(args.data_dir, class_names)
    
    
    # First split: 80% train+val, 20% test
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=args.seed, stratify=labels
    )
    # Second split: 87.5% train, 12.5% val from the 80% train+val
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, test_size=0.125, random_state=args.seed, 
        stratify=train_val_labels
    )
    
    class_weights = calculate_class_weights(train_labels, len(class_names))
    print("Class distribution in training set:")
    train_counts = Counter(train_labels)
    for idx, name in enumerate(sorted(class_names)):
        count = train_counts.get(class_to_idx[name], 0)
        weight = class_weights[idx].item()
        print(f"  {name}: {count} samples (weight: {weight:.3f})")
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Datasets
    train_dataset = AnimalDataset(train_paths, train_labels, train_transform)
    val_dataset = AnimalDataset(val_paths, val_labels, val_transform)
    test_dataset = AnimalDataset(test_paths, test_labels, val_transform)
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    
    # Model
    print("\n" + "=" * 80)
    print("CREATING MODEL")
    print("=" * 80)
    model = create_model(len(class_names), pretrained=True)
    model = model.to(device)
    
    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    print("✓ Using weighted CrossEntropyLoss for class imbalance")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )
    
    # Training loop
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)
    
    best_val_acc = 0
    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        print("-" * 80)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"✓ New best validation accuracy! Saving model...")
            
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_to_idx': class_to_idx,
                'idx_to_class': {v: k for k, v in class_to_idx.items()}
            }, os.path.join(args.output_dir, 'best_model.pth'))
            
            with open(os.path.join(args.output_dir, 'class_mapping.json'), 'w') as f:
                json.dump({
                    'class_to_idx': class_to_idx,
                    'idx_to_class': {v: k for k, v in class_to_idx.items()}
                }, f, indent=2)
    
    print("\n" + "=" * 80)
    print("TESTING ON HELD-OUT TEST SET")
    print("=" * 80)
    
    # Load best model
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
    
    # Save training history plot
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.axhline(y=test_acc, color='r', linestyle='--', label=f'Test Acc ({test_acc:.2f}%)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'training_history.png'), dpi=300)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    print(f"Model saved to: {args.output_dir}")
    print("=" * 80)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train animal classification model')
    parser.add_argument('--data_dir', type=str, default='raw-img')
    parser.add_argument('--output_dir', type=str, default='models/image_classifier')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    train_classifier(args)