import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm

from glob import glob

folder_content = glob("Data/D_and_C/train/*.jpg")

paths = []
labels = []
for item in folder_content:
    item_split = item.split("/")[-1].split(".")
    paths.append(item)
    if item_split[0] == 'dog':
        labels.append(0)
    if item_split[0] == 'cat':
        labels.append(1)

import os
import pandas as pd
from torchvision.io import decode_image
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.img_labels = labels
        self.img_dir = paths
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = f"{self.img_dir[idx]}"
        image = Image.open(img_path).convert("RGB")
        label = torch.tensor(int(self.img_labels[idx]))
        if self.transform:
            image = self.transform(image)
        return image, label
    

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
c_and_d_dataset = CustomImageDataset(paths, labels, transform=transform)
c_and_d_dataloader = DataLoader(c_and_d_dataset, batch_size=64, shuffle=True)

def getTransformer(transform_resize, transform_crop, transform_normalize_mean, transform_normalize_var):

    transform = transforms.Compose(
            [
                transforms.Resize(transform_resize),
                transforms.RandomCrop(transform_crop),
                transforms.RandomRotation(90),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(transform_normalize_mean, transform_normalize_var),
            ]
        )

    return transform

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Define a simple CNN for binary classification
class BinaryCNN(nn.Module):
    def __init__(self):
        super(BinaryCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 128),  # Assumes input images are resized to 224x224
            nn.ReLU(),
            nn.Linear(128, 1)  # Output single logit for binary classification
        )

    def forward(self, x):
        return self.net(x)
    

def train_model(model, dataloader, criterion, optimizer, device, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)  # labels: [batch_size, 1]

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = torch.sigmoid(outputs) > 0.5
            correct += (preds.float() == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BinaryCNN().to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Assuming train_loader is already defined
train_model(model, c_and_d_dataloader, criterion, optimizer, device, epochs=100)
