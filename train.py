import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from tqdm import tqdm

# ==========================================
# 1. Configuration & Hyperparameters
# ==========================================
DATA_DIR = "./PlantVillage"  # Points directly to D:\plant-disease-rag\plant-disease-rag\PlantVillage
MODEL_SAVE_PATH = "./models/plant_disease_efficientnet.pth"
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[INFO] Using compute device: {DEVICE}")

# Ensure models output directory exists
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

# ==========================================
# 2. Data Transforms & Loaders
# ==========================================
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Check dataset directory
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"[ERROR] Could not find dataset folder at '{DATA_DIR}'. Check folder path!")

full_dataset = datasets.ImageFolder(root=DATA_DIR)
class_names = full_dataset.classes

print(f"[INFO] Total dataset size: {len(full_dataset)} images")
print(f"[INFO] Number of disease classes detected: {len(class_names)}")

# Split into Train (80%) and Validation (20%)
val_size = int(len(full_dataset) * VAL_SPLIT)
train_size = len(full_dataset) - val_size

train_dataset, val_dataset = random_split(
    full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
)

train_dataset.dataset.transform = train_transforms
val_dataset.dataset.transform = val_transforms

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ==========================================
# 3. Build & Initialize EfficientNet-B0
# ==========================================
def build_model(num_classes):
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)

    # Freeze base feature layers
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace classification head with target class count
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, num_classes)
    )
    return model

model = build_model(len(class_names)).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

# ==========================================
# 4. Training Loop
# ==========================================
def train_model(model, criterion, optimizer, scheduler, num_epochs):
    start_time = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")

        # Training Phase
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        print(f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc * 100:.2f}%")

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        epoch_val_loss = val_loss / len(val_dataset)
        epoch_val_acc = val_corrects.double() / len(val_dataset)
        print(f"Val Loss:   {epoch_val_loss:.4f} | Val Acc:   {epoch_val_acc * 100:.2f}%")

        scheduler.step(epoch_val_loss)

        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"[SUCCESS] Saved new best model weights to {MODEL_SAVE_PATH}")

    time_elapsed = time.time() - start_time
    print(f"\n[COMPLETE] Training finished in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"[RESULTS] Highest Validation Accuracy: {best_acc * 100:.2f}%")

    model.load_state_dict(best_model_wts)
    return model

if __name__ == "__main__":
    train_model(model, criterion, optimizer, scheduler, NUM_EPOCHS)