import os
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0

def convert_specific_pth_to_onnx():
    pth_path = os.path.join("models", "plant_disease2_efficientnet.pth")
    onnx_path = os.path.join("models", "plant_disease2_efficientnet.onnx")

    print(f"[1/4] Checking model file at '{pth_path}'...")
    if not os.path.exists(pth_path):
        print(f"? Error: Could not find '{pth_path}'. Ensure the path is correct.")
        return

    device = torch.device("cpu")
    checkpoint = torch.load(pth_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        class_names = checkpoint.get("class_names", [f"Class_{i}" for i in range(15)])
    else:
        state_dict = checkpoint
        class_names = [f"Class_{i}" for i in range(15)]

    num_classes = len(class_names)
    print(f"[2/4] Checkpoint loaded. Detected {num_classes} classes.")

    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    
    if any(k.startswith("classifier.1.1.") for k in state_dict.keys()):
        model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )
    else:
        model.classifier[1] = nn.Linear(in_features, num_classes)

    model.load_state_dict(state_dict)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224, device=device)

    print(f"[3/4] Exporting to '{onnx_path}'...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )

    print(f"? [SUCCESS] Created ONNX model at: {onnx_path}")

if __name__ == "__main__":
    convert_specific_pth_to_onnx()
