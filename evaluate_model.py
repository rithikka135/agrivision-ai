import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from app.vision import classifier

def evaluate_and_store_metrics(test_dir="dataset/val", output_dir="data"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)
    print(f"[EVALUATION] Starting evaluation on device: {device}")

    if not os.path.exists(test_dir):
        print(f"[ERROR] Test dataset folder '{test_dir}' not found.")
        return

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
    
    # Safe Windows DataLoader (num_workers=0 avoids RAM multiplication)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    model = classifier.model.to(device)
    model.eval()

    y_true, y_pred = [], []

    print(f"[EVALUATION] Running inference on {len(test_dataset)} real validation images...")
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            
            # Print progress every 50 batches
            if (i + 1) % 50 == 0 or (i + 1) == len(test_loader):
                print(f"Processed Batch {i + 1}/{len(test_loader)} ({(i + 1) * 32} / {len(test_dataset)} images)")

    class_names = test_dataset.classes

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

    # Save CSV
    csv_path = os.path.join(output_dir, "per_class_metrics.csv")
    pd.DataFrame(report_dict).transpose().to_csv(csv_path)

    # Save Confusion Matrix Heatmap
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(18, 14))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('AgriVision AI - Real Plant Disease Confusion Matrix', fontsize=14)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)

    print("\n" + "="*50)
    print("        REAL DATASET EVALUATION COMPLETE        ")
    print("="*50)
    print(f"Real Overall Accuracy : {acc * 100:.2f}%")
    print(f"Weighted Precision    : {precision:.4f}")
    print(f"Weighted Recall       : {recall:.4f}")
    print(f"Weighted F1-Score     : {f1:.4f}")
    print("="*50)
    print(f"Saved Reports to: {output_dir}/")

if __name__ == "__main__":
    evaluate_and_store_metrics(test_dir="dataset/val")