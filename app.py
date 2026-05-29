# Retinal Disease Classification Demo
# MSB7216: Deep Learning for Health Data
# Gradio interface for single image inference with Grad-CAM overlay
#
# Best model: ConvNeXt-Tiny + Weighted Cross-Entropy
# Dataset: Eye Disease Image Dataset (Sharmin et al. 2024)
# Hosted on Hugging Face Spaces

# Retinal Disease Classification Demo
# MSB7216: Deep Learning for Health Data
# Gradio interface with Grad-CAM

import gradio as gr
import torch
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms
import timm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import warnings
warnings.filterwarnings('ignore')


# Configuration
MODEL_PATH  = 'model.safetensors'   # ← Updated for safetensors
DEVICE      = torch.device('cpu')
IMG_SIZE    = 224

CLASS_NAMES = [
    'Diabetic Retinopathy', 'Glaucoma', 'Healthy', 'Myopia', 'Macular Scar',
    'Retinitis Pigmentosa', 'Disc Edema', 'Retinal Detachment',
    'Central Serous Chorioretinopathy', 'Pterygium'
]

# Clinical urgency and descriptions (unchanged)
URGENCY = { ... }          # Keep your existing URGENCY dictionary
DESCRIPTIONS = { ... }     # Keep your existing DESCRIPTIONS dictionary


# Model loading - Updated for safetensors
def load_model():
    model = timm.create_model('convnext_tiny', pretrained=False, num_classes=len(CLASS_NAMES))
    
    # Load safetensors format
    from safetensors.torch import load_file
    state_dict = load_file(MODEL_PATH)
    model.load_state_dict(state_dict)
    
    model = model.to(DEVICE).eval()
    print(f"Model loaded successfully from {MODEL_PATH}")
    return model


model = load_model()


# Preprocessing (unchanged)
def preprocess_fundus(img_pil: Image.Image) -> np.ndarray:
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    _, green, _ = cv2.split(img_bgr)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    green_clahe = clahe.apply(green)
    img_3ch = cv2.merge([green_clahe, green_clahe, green_clahe])
    img_resized = cv2.resize(img_3ch, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    return img_resized


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# Grad-CAM (unchanged)
class GradCAMExtractor:
    def __init__(self, model, target_layer_name='stages.3'):
        self.model = model
        self.activations = None
        self.gradients = None
        self._register_hooks(target_layer_name)

    def _register_hooks(self, layer_name):
        for name, module in self.model.named_modules():
            if name == layer_name:
                module.register_forward_hook(self._save_activation)
                module.register_full_backward_hook(self._save_gradient)
                return
        raise ValueError(f'Layer {layer_name} not found.')

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad()
        output = self.model(tensor)
        output[0, class_idx].backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


gradcam = GradCAMExtractor(model, target_layer_name='stages.3')


# ... (keep your make_gradcam_overlay and predict functions as they were) ...

def make_gradcam_overlay(img_np: np.ndarray, cam: np.ndarray) -> Image.Image:
    cam_resized = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    heatmap = (cm.jet(cam_resized)[:, :, :3] * 255).astype(np.uint8)
    overlay = cv2.addWeighted(img_np, 0.5, heatmap, 0.5, 0)
    return Image.fromarray(overlay)


# Inference function
def predict(image: Image.Image):
    if image is None:
        return "Please upload a fundus image.", None, {}, "No image provided."

    try:
        img_np = preprocess_fundus(image)
        img_pil = Image.fromarray(img_np)
        tensor = eval_transform(img_pil).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1)[0]

        pred_idx = probs.argmax().item()
        confidence = probs[pred_idx].item()
        pred_class = CLASS_NAMES[pred_idx]

        # Grad-CAM
        tensor_grad = eval_transform(img_pil).unsqueeze(0).to(DEVICE)
        tensor_grad.requires_grad_(True)
        cam = gradcam(tensor_grad, pred_idx)
        overlay = make_gradcam_overlay(img_np, cam)

        all_probs = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

        urgency = URGENCY[pred_class]
        desc = DESCRIPTIONS[pred_class]

        result_text = f"""Predicted: {pred_class}
Confidence: {confidence:.1%}

Clinical Urgency: {urgency}

About this condition:
{desc}

DISCLAIMER: This is a research prototype only. Not for clinical use."""

        return result_text, overlay, all_probs, result_text

    except Exception as e:
        error_msg = f"Error during inference: {str(e)}"
        return error_msg, None, {}, error_msg

# Interface
TITLE = "Retinal Disease Classification from Fundus Images"

DESCRIPTION = """
**MSB7216: Deep Learning for Health Data | Final Examination Project**

This demo classifies retinal fundus photographs into 10 disease categories using a 
**ConvNeXt-Tiny** model fine-tuned on the Eye Disease Image Dataset 
(Sharmin et al. 2024, collected from two ophthalmology hospitals in Bangladesh).

The Grad-CAM overlay highlights which regions of the image influenced the prediction.

**Classes:** Diabetic Retinopathy, Glaucoma, Healthy, Myopia, Macular Scar, 
Retinitis Pigmentosa, Disc Edema, Retinal Detachment, Central Serous Chorioretinopathy, Pterygium.

> **Disclaimer:** Research prototype only. Not validated for clinical use.
"""

ARTICLE = """
### Model Details
| Property | Value |
|:---|:---|
| Architecture | ConvNeXt-Tiny (Liu et al. 2022) |
| Pretrained on | ImageNet |
| Fine-tuned on | Eye Disease Image Dataset (5,335 images, 10 classes) |
| Loss function | Weighted Cross-Entropy |
| Test Accuracy | 0.7553 |
| Test Macro F1 | 0.7783 |
| Preprocessing | Green channel extraction + CLAHE |

### Dataset
Sharmin et al. (2024). Eye Disease Image Dataset. Mendeley Data.  
Source: Anwara Hamida Eye Hospital and B.N.S.B. Zahurul Haque Eye Hospital, Faridpur, Bangladesh.  
License: CC BY 4.0

### References
- Liu et al. (2022). A ConvNet for the 2020s. CVPR.
- Selvaraju et al. (2017). Grad-CAM. ICCV.
"""

with gr.Blocks(title=TITLE, theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# {TITLE}")
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type='pil',
                label='Upload Fundus Image',
                height=300
            )
            submit_btn = gr.Button('Classify', variant='primary')

        with gr.Column(scale=1):
            gradcam_output = gr.Image(
                type='pil',
                label='Grad-CAM Activation Map',
                height=300
            )

    with gr.Row():
        with gr.Column(scale=1):
            result_text = gr.Textbox(
                label='Prediction and Clinical Context',
                lines=10,
                interactive=False
            )
        with gr.Column(scale=1):
            prob_output = gr.Label(
                label='Class Probabilities',
                num_top_classes=5
            )

    submit_btn.click(
        fn=predict,
        inputs=image_input,
        outputs=[result_text, gradcam_output, prob_output, result_text]
    )

    gr.Markdown(ARTICLE)

if __name__ == '__main__':
    demo.launch()
