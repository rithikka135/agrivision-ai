import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from app.vision import classifier

def generate_synthetic_test_data(data_dir="data/test_sample", num_samples_per_class=3):
    """Generates dummy sample images for testing model evaluation pipeline."""
    print("[TEST SETUP] Creating test image dataset structure...")
    os.makedirs(data_dir, exist_ok=True)
    
    # Use class names loaded in your classifier
    classes = classifier.class_names
    
    for cls_name in classes:
        cls_folder = os.path.join(data_dir, cls_name)
        os.makedirs(cls_folder, exist_ok=True)
        
        for i in range(num_samples_per_class):
            img_path = os.path.join(cls_folder, f"sample_{i}.jpg")
            if not os.path.exists(img_path):
                # Create a 224x224 RGB image with random noise
                random_img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
                img = Image.fromarray(random_img)
                img.save(img_path)
                
    print(f"[TEST SETUP] Created sample dataset in '{data_dir}' across {len(classes)} classes.")

def evaluate_and_store_metrics(test_dir="data/test_sample", output_dir="data"):
    """Runs evaluation on the dataset and exports JSON, CSV, and Confusion Matrix."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = classifier.model.to(device)
    model.eval()

    y_true, y_pred = [], []

    print(f"[EVALUATION] Starting model evaluation on {len(test_dataset)} images...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    class_names = test_dataset.classes

    # Compute overall metrics
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)

    # Save JSON summary
    json_path = os.path.join(output_dir, "evaluation_summary.json")
    summary_data = {
        "overall_accuracy": round(float(acc) * 100, 2),
        "weighted_precision": round(float(precision), 4),
        "weighted_recall": round(float(recall), 4),
        "weighted_f1_score": round(float(f1), 4),
        "per_class_report": report_dict
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4)

    # Save CSV metrics
    csv_path = os.path.join(output_dir, "per_class_metrics.csv")
    pd.DataFrame(report_dict).transpose().to_csv(csv_path)

    # Save Confusion Matrix Heatmap
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(16, 12))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('AgriVision AI - Plant Disease Confusion Matrix')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)

    print("\n" + "="*50)
    print("           EVALUATION COMPLETE & STORED           ")
    print("="*50)
    print(f"Overall Accuracy : {acc * 100:.2f}%")
    print(f"Saved JSON Report : {json_path}")
    print(f"Saved CSV Report  : {csv_path}")
    print(f"Saved Plot        : {cm_path}")
    print("="*50)

if __name__ == "__main__":
    test_folder = "data/test_sample"
    generate_synthetic_test_data(data_dir=test_folder)
    evaluate_and_store_metrics(test_dir=test_folder)