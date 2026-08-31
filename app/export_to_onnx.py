import os
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0

def convert_pth_to_onnx(
    pth_path="models/plant_disease2_efficientnet.pth",
    onnx_path="models/plant_disease_efficientnet.onnx"
):
    print(f"[1/4] Checking model file at '{pth_path}'...")
    if not os.path.exists(pth_path):
        print(f"❌ Error: Could not find {pth_path}. Please check your path.")
        return

    device = torch.device("cpu")  # Exporting on CPU ensures target portability

    # [2/4] Load PyTorch Checkpoint
    checkpoint = torch.load(pth_path, map_location=device)

    # Determine class names and number of classes from checkpoint or default
    if isinstance(checkpoint, dict) and 'class_names' in checkpoint:
        class_names = checkpoint['class_names']
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        # Default fallback class count if not embedded in checkpoint dictionary
        class_names = [f"Class_{i}" for i in range(15)] 

    num_classes = len(class_names)
    print(f"[2/4] Detected {num_classes} classes.")

    # [3/4] Reconstruct Model Architecture
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    # Load state dict
    model.load_state_dict(state_dict)
    model.eval()

    # [4/4] Export to ONNX
    # Create a dummy tensor matching ImageNet input shape: (Batch Size, Channels, Height, Width)
    dummy_input = torch.randn(1, 3, 224, 224, device=device)

    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    print(f"[3/4] Exporting computation graph to '{onnx_path}'...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,             # Store trained parameters inside the ONNX file
        opset_version=14,               # Stable ONNX operator set version
        do_constant_folding=True,       # Optimize constant nodes for speed
        input_names=['input'],          # Input layer name
        output_names=['output'],        # Output layer name
        dynamic_axes={                  # Support variable batch sizes
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"✅ [SUCCESS] Saved ONNX model to: {onnx_path}")

if __name__ == "__main__":
    convert_pth_to_onnx()