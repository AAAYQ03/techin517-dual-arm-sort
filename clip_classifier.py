#!/usr/bin/env python3
"""CLIP zero-shot 二次分类 - 区分 battery vs usb stick vs 机械臂"""
import cv2, torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

class ClipClassifier:
    CLASSES = ['pen', 'glue stick', 'battery', 'earbuds case', 'robotic arm', 'empty background']
    TEMPLATES = ["a photo of a {}", "a close-up of a {}", "a {} on a desk"]

    def __init__(self, device='cuda', model_name="openai/clip-vit-base-patch32"):
        self.device = device
        print(f"[CLIP] Loading {model_name}...")
        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        text_in = [t.format(c) for c in self.CLASSES for t in self.TEMPLATES]
        ti = self.processor(text=text_in, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            tf = self.model.get_text_features(**ti)
            tf = tf / tf.norm(dim=-1, keepdim=True)
            tf = tf.view(len(self.CLASSES), len(self.TEMPLATES), -1).mean(dim=1)
            self.text_features = tf / tf.norm(dim=-1, keepdim=True)
        print(f"[CLIP] Ready ({len(self.CLASSES)} classes)")

    def classify(self, image_bgr, bbox, pad=15):
        """对 bbox crop 跑 CLIP. 返回 (label, confidence, all_scores)"""
        H, W = image_bgr.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cx1, cy1 = max(0, x1-pad), max(0, y1-pad)
        cx2, cy2 = min(W, x2+pad), min(H, y2+pad)
        crop = image_bgr[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return 'empty background', 0.0, {}
        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        ii = self.processor(images=pil, return_tensors="pt").to(self.device)
        with torch.no_grad():
            f = self.model.get_image_features(**ii)
            f = f / f.norm(dim=-1, keepdim=True)
            s = (self.model.logit_scale.exp() * (f @ self.text_features.T)).softmax(-1)[0]
        idx = s.argmax().item()
        scores = {self.CLASSES[i]: s[i].item() for i in range(len(self.CLASSES))}
        return self.CLASSES[idx], s[idx].item(), scores
