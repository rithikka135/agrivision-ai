import os
import shutil
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import kagglehub

def main():
    print("[1/3] Downloading PlantVillage dataset via kagglehub...")
    downloaded_path = kagglehub.dataset_download("abdallahalidev/plantvillage-dataset")

    color_dir = os.path.join(downloaded_path, "plantvillage dataset", "color")
    if not os.path.exists(color_dir):
        for root, dirs, _ in os.walk(downloaded_path):
            if "color" in dirs:
                color_dir = os.path.join(root, "color")
                break

    DATASET_PATH = "dataset"
    train_dir = os.path.join(DATASET_PATH, "train")
    val_dir = os.path.join(DATASET_PATH, "val")

    if not os.path.exists(train_dir) or len(os.listdir(train_dir)) == 0:
        print("[2/3] Structuring dataset into train/val splits...")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        classes = [d for d in os.listdir(color_dir) if os.path.isdir(os.path.join(color_dir, d))]

        for cls in classes:
            cls_src = os.path.join(color_dir, cls)
            cls_train = os.path.join(train_dir, cls)
            cls_val = os.path.join(val_dir, cls)

            os.makedirs(cls_train, exist_ok=True)
            os.makedirs(cls_val, exist_ok=True)

            images = [f for f in os.listdir(cls_src) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            random.shuffle(images)

            split_idx = int(0.8 * len(images))
            train_imgs = images[:split_idx]
            val_imgs = images[split_idx:]

            for img in train_imgs:
                shutil.copy(os.path.join(cls_src, img), os.path.join(cls_train, img))
            for img in val_imgs:
                shutil.copy(os.path.join(cls_src, img), os.path.join(cls_val, img))
    else:
        print("[2/3] Existing dataset directory found, skipping split creation.")

    print("[3/3] Initializing PyTorch training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(DATASET_PATH, x), data_transforms[x])
        for x in ['train', 'val']
    }

    # Increased batch size to 64 for faster CPU iteration
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=64, shuffle=True, num_workers=0)
        for x in ['train', 'val']
    }

    class_names = image_datasets['train'].classes
    num_classes = len(class_names)
    print(f"Detected {num_classes} plant disease classes!")

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_ftrs, num_classes)
    )

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    epochs = 3  # Reduced to 3 epochs for quick training
    best_acc = 0.0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 35)

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()

            running_loss = 0.0
            running_corrects = 0
            total_batches = len(dataloaders[phase])

            for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # Real-Time Progress Tracker every 10 batches
                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                    print(f"[{phase.upper()}] Batch {batch_idx + 1}/{total_batches} processed...", end="\r")

            print() # New line after phase completion
            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])
            print(f"--> {phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                os.makedirs("models", exist_ok=True)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'class_names': class_names
                }, 'models/plant_disease_efficientnet.pth')
                print(f"*** [SAVED] Weights updated: models/plant_disease_efficientnet.pth (Val Acc: {best_acc:.4f}) ***")

    print("\n[COMPLETE] Model training successfully finished!")

if __name__ == '__main__':
    main()