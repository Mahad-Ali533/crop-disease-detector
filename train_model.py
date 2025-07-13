import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
from PIL import Image

# === CONFIG ===
DATA_DIR = r"C:\Users\san\Desktop\crop_disease_detector\dataset"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5

# === STEP 1: Normalize file extensions ===
def normalize_image_extensions(folder):
    count = 0
    for root, _, files in os.walk(folder):
        for file in files:
            ext = os.path.splitext(file)[1]
            if ext.lower() in ['.jpg', '.jpeg', '.png'] and ext != ext.lower():
                old_path = os.path.join(root, file)
                new_name = os.path.splitext(file)[0] + ext.lower()
                new_path = os.path.join(root, new_name)
                os.rename(old_path, new_path)
                count += 1
    print(f"✅ Normalized {count} image file extensions to lowercase.")

normalize_image_extensions(DATA_DIR)

# === STEP 3: Transforms ===
train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

val_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

# === STEP 4: Load Data ===
print("📦 Loading data from:", TRAIN_DIR)
train_data = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
print("📦 Loading validation data from:", VAL_DIR)
val_data = datasets.ImageFolder(VAL_DIR, transform=val_transforms)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

print("📚 Classes found:", train_data.classes)

# === STEP 5: Define Model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.last_channel, len(train_data.classes))
model = model.to(device)

# === STEP 6: Train Model ===
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("\n🚀 Starting training...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"📈 Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss / len(train_loader):.4f}")

# === STEP 7: Save Model ===
model_dir = "app/model"
os.makedirs(model_dir, exist_ok=True)
torch.save(model.state_dict(), os.path.join(model_dir, "model.pt"))
print("✅ Model saved to app/model/model.pt")

# === STEP 8: File Count Summary ===
print("\n📊 Sample file count per class:")
for folder in os.listdir(TRAIN_DIR):
    folder_path = os.path.join(TRAIN_DIR, folder)
    if os.path.isdir(folder_path):
        images = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        print(f"{folder}: {len(images)} images — e.g. {images[:3]}")
