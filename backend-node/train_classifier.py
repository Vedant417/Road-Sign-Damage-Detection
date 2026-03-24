import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image

# Configuration
NUM_EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
DATA_DIR = '../Signs/DATA'
MODEL_SAVE_PATH = 'sign_classifier_resnet50.pth'

class GTSRBDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Determine number of classes
        self.classes = sorted([int(d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        for class_id in self.classes:
            class_dir = os.path.join(data_dir, str(class_id))
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(class_id)
                    
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def main():
    print("Initializing ResNet50 Transfer Learning Pipeline...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Advanced Data Augmentations (Including Perspective Distortion)
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.5), # Simulate camera angles
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(p=0.5), # CRITICAL: Ensure Left/Right datasets are properly managed for this!
        transforms.ColorJitter(brightness=0.3, contrast=0.3), 
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)), # Simulate bad focal length
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("Loading datasets...")
    full_dataset = GTSRBDataset(DATA_DIR)
    num_classes = len(full_dataset.classes)
    
    # 80/20 Train-Val split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_dataset.dataset.transform = train_transforms
    val_dataset.dataset.transform = val_transforms
    
    # Class balancing through proper batch loading is recommended, but shuffle handles basic distribution
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Load Pretrained ResNet50 instead of MobileNet for higher capacity
    model = models.resnet50(pretrained=True)
    
    # Unfreeze all layers for complete fine-tuning on our sign features
    for param in model.parameters():
        param.requires_grad = True
        
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    # Use AdamW for better weight decay properties on deep ResNets
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    print("Starting Training Loop...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0; total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_acc = 100 * correct / total
        
        model.eval()
        val_correct = 0; val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_acc = 100 * val_correct / val_total
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Loss: {running_loss/len(train_loader):.4f} - Train Acc: {train_acc:.2f}% - Val Acc: {val_acc:.2f}%")
        
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Training Complete! ResNet50 classifier saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
