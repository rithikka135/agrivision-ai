import os
import torch

def inspect_active_model():
    # Paths to both candidate model files
    model1_path = os.path.join("models", "plant_disease_efficientnet.pth")
    model2_path = os.path.join("models", "plant_disease2_efficientnet.pth")
    
    print("\n==================================================")
    print("🔍 CHECKING AVAILABLE DEEP LEARNING MODEL FILES")
    print("==================================================")
    
    # Check Model 1
    if os.path.exists(model1_path):
        print(f"✅ Found Model 1: {model1_path}")
        try:
            ckpt1 = torch.load(model1_path, map_location="cpu")
            if isinstance(ckpt1, dict) and 'class_names' in ckpt1:
                print(f"   └── Classes: {len(ckpt1['class_names'])} classes detected")
            else:
                print("   └── Format: Raw PyTorch State Dict")
        except Exception as e:
            print(f"   └── Error reading file: {e}")
    else:
        print(f"❌ Model 1 Not Found: {model1_path}")

    # Check Model 2
    if os.path.exists(model2_path):
        print(f"\n✅ Found Model 2: {model2_path}")
        try:
            ckpt2 = torch.load(model2_path, map_location="cpu")
            if isinstance(ckpt2, dict) and 'class_names' in ckpt2:
                print(f"   └── Classes: {len(ckpt2['class_names'])} classes detected")
            else:
                print("   └── Format: Raw PyTorch State Dict")
        except Exception as e:
            print(f"   └── Error reading file: {e}")
    else:
        print(f"❌ Model 2 Not Found: {model2_path}")

    print("\n==================================================")
    print("🚀 DETERMINING ACTIVE INFERENCE ENGINE MODEL")
    print("==================================================")

    # Replicate vision.py selection logic
    selected_model = model1_path
    if not os.path.exists(model1_path) and os.path.exists(model2_path):
        selected_model = model2_path
    elif os.path.exists(model2_path):
        selected_model = model2_path  # Prefers Model 2 if both exist

    if os.path.exists(selected_model):
        abs_path = os.path.abspath(selected_model)
        print(f"🎯 ACTIVE MODEL LOADED BY APP: {selected_model}")
        print(f"📍 Full Path: {abs_path}")
    else:
        print("⚠️ WARNING: No valid .pth checkpoint found in models/ directory!")
    print("==================================================\n")

if __name__ == "__main__":
    inspect_active_model()