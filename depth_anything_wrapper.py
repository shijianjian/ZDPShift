import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


class DepthAnything:
    def __init__(self, size="large", device="cuda"):
        hf_id = f"depth-anything/Depth-Anything-V2-{size.capitalize()}-hf"
        self.processor = AutoImageProcessor.from_pretrained(hf_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(hf_id).to(device).eval()
        self.device = device

    @torch.no_grad()
    def predict(self, image):
        """image: (3, H, W) float32 [0, 1]. Returns (1, H, W) relative depth (larger = closer)."""
        H, W = image.shape[-2:]
        pil = Image.fromarray((image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
        inputs = self.processor(images=pil, return_tensors="pt").to(self.device)
        depth = self.model(**inputs).predicted_depth   # (1, H', W')
        depth = F.interpolate(depth.unsqueeze(1), (H, W), mode="bilinear", align_corners=True)
        return depth.squeeze(0).float()                # (1, H, W)
