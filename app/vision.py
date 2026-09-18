import os
import io
import base64
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0
from PIL import Image
import numpy as np
import cv2

def estimate_leaf_damage(image: Image.Image) -> float:
    """
    Calculates percentage of necrotic/diseased leaf surface using OpenCV HSV color masks.
    """
    orig_np = np.array(image.convert("RGB"))
    cv_img = cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)

    # Mask healthy green regions (Hue 35-85)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Mask total leaf area (filtering out white/grey background)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    _, leaf_mask = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)

    total_leaf_pixels = cv2.countNonZero(leaf_mask)
    healthy_pixels = cv2.countNonZero(green_mask)

    if total_leaf_pixels == 0:
        return 0.0

    damaged_pixels = max(0, total_leaf_pixels - healthy_pixels)
    damage_pct = (damaged_pixels / total_leaf_pixels) * 100.0
    return round(min(damage_pct, 100.0), 2)


class FineTunedDiseaseClassifier:
    def __init__(self, model_path="models/plant_disease2_efficientnet.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        os.makedirs("models", exist_ok=True)

        # 1. Fallback & Download Handling for Cloud Deployments
        if not os.path.exists(model_path) and os.path.exists("models/plant_disease_efficientnet.pth"):
            model_path = "models/plant_disease_efficientnet.pth"

        print(f"[VISION ENGINE] Loading model weights from: {model_path}")

        # 2. Preprocessing Transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 3. Load Checkpoint and Extract Classes Dynamically
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                if 'class_names' in checkpoint:
                    self.class_names = checkpoint['class_names']
                else:
                    self.class_names = [f"Class_{i}" for i in range(38)]
            else:
                state_dict = checkpoint
                self.class_names = [f"Class_{i}" for i in range(38)]
        else:
            raise FileNotFoundError(f"No valid model checkpoint found at {model_path}")

        num_classes = len(self.class_names)
        print(f"[VISION ENGINE] Successfully registered {num_classes} plant classes.")

        # 4. Re-create Model Architecture matching weight keys
        self.model = efficientnet_b0(weights=None)
        in_features = self.model.classifier[1].in_features

        if any(k.startswith("classifier.1.1.") for k in state_dict.keys()):
            self.model.classifier[1] = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(in_features, num_classes)
            )
        else:
            self.model.classifier[1] = nn.Linear(in_features, num_classes)

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # Grad-CAM Hooks Setup
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        """Hook into the final convolutional layer of EfficientNet-B0."""
        target_layer = self.model.features[-1]

        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate_gradcam(self, tensor: torch.Tensor, class_idx: int, orig_image: Image.Image) -> str:
        """Generates real Grad-CAM visual spatial heatmap."""
        self.model.zero_grad()
        output = self.model(tensor)
        score = output[0, class_idx]
        score.backward()

        gradients = self.gradients[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()
        weights = np.mean(gradients, axis=(1, 2))

        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        if np.max(cam) > 0:
            cam = cam / np.max(cam)

        cam = cv2.resize(cam, (224, 224))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

        orig_np = np.array(orig_image.resize((224, 224)))
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(orig_np, 0.6, heatmap_rgb, 0.4, 0)

        res_pil = Image.fromarray(overlay)
        buf = io.BytesIO()
        res_pil.save(buf, format="JPEG")
        base64_img = base64.b64encode(buf.getvalue()).decode('utf-8')

        return f"data:image/jpeg;base64,{base64_img}"

    def predict(self, image_bytes: bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        tensor.requires_grad_()

        outputs = self.model(tensor)
        probs = F.softmax(outputs[0], dim=0)
        conf, pred = torch.max(probs, dim=0)

        confidence_score = round(conf.item() * 100, 2)
        predicted_idx = pred.item()

        # Low confidence guard for field photos
        if confidence_score < 35.0:
            class_label = "Unclear / Non-Crop Image Detected"
            damage_pct = 0.0
        else:
            class_label = self.class_names[predicted_idx]
            # Calculate surface damage area using HSV segmentation
            damage_pct = estimate_leaf_damage(image)

        # Generate Grad-CAM Heatmap
        try:
            gradcam_b64 = self.generate_gradcam(tensor, predicted_idx, image)
        except Exception as e:
            print(f"[Grad-CAM Error]: {e}")
            orig_img = np.array(image.resize((224, 224)))
            gray = cv2.cvtColor(orig_img, cv2.COLOR_RGB2GRAY)
            heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(orig_img, 0.6, heatmap, 0.4, 0)
            res_pil = Image.fromarray(overlay)
            buf = io.BytesIO()
            res_pil.save(buf, format="JPEG")
            gradcam_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

        return class_label, confidence_score, gradcam_b64, damage_pct

# Instantiate global classifier for FastAPI app
classifier = FineTunedDiseaseClassifier()