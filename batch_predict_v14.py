"""
batch_predict.py  —  v14  (configurable backbone + TTA)

Usage:
    python batch_predict.py
    python batch_predict.py --backbone mobilenetv3_large_100
    python batch_predict.py --backbone resnet50 --model_dir result/food_model_resnet50
    python batch_predict.py --tta 1   (disable TTA for speed)
"""

import argparse
import os
import glob
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image as PILImage
import timm

DEFAULT_BACKBONE  = 'efficientnet_b3'
DEFAULT_MODEL_DIR = f'result/food_model_{DEFAULT_BACKBONE}'
DEFAULT_CSV       = 'data_from_intragram_augmented2.csv'
DEFAULT_IMGDIR    = 'Intragram Images [Original]/'
IMG_SIZE          = 224
TEMPERATURE       = 1.5
N_TTA             = 5

parser = argparse.ArgumentParser()
parser.add_argument('--backbone',  default=DEFAULT_BACKBONE)
parser.add_argument('--model_dir', default=None,
                    help='Auto-set from backbone if not specified')
parser.add_argument('--pred',      default=DEFAULT_CSV)
parser.add_argument('--img_dir',   default=DEFAULT_IMGDIR)
parser.add_argument('--tta',       type=int, default=N_TTA)
args = parser.parse_args()

if args.model_dir is None:
    args.model_dir = f'result/food_model_{args.backbone}'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device  : {DEVICE}")
print(f"Backbone: {args.backbone}")
print(f"Models  : {args.model_dir}")
print(f"TTA     : {args.tta} views")

# ── MODEL ─────────────────────────────────────────────────────────────
class SiameseNet(nn.Module):
    def __init__(self, backbone_name):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=False, num_classes=0
        )
        feat_dim = self.backbone.num_features
        cmp_dim  = feat_dim * 2
        hidden   = max(128, cmp_dim // 10)
        self.comparison_head = nn.Sequential(
            nn.Linear(cmp_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.45),
            nn.Linear(hidden, 1),
        )
        self.temperature = TEMPERATURE

    def forward(self, img_a, img_b):
        self.backbone.eval()
        with torch.no_grad():
            fa = self.backbone(img_a)
            fb = self.backbone(img_b)
        fa = F.normalize(fa, dim=1)
        fb = F.normalize(fb, dim=1)
        diff     = fa - fb
        prod     = fa * fb
        combined = torch.cat([diff, prod], dim=1)
        logit    = self.comparison_head(combined).squeeze(1)
        return torch.sigmoid(logit / self.temperature)

# ── LOAD ENSEMBLE ─────────────────────────────────────────────────────
model_paths = sorted(glob.glob(os.path.join(args.model_dir, 'fold*_best.pth')))
if not model_paths:
    model_paths = [os.path.join(args.model_dir, 'final_model.pth')]
if not model_paths or not os.path.exists(model_paths[0]):
    raise FileNotFoundError(f"No models in '{args.model_dir}'. Train first.")

print(f"Loading {len(model_paths)} model(s)...")
models = []
for p in model_paths:
    m = SiameseNet(args.backbone).to(DEVICE)
    m.load_state_dict(torch.load(p, map_location=DEVICE))
    m.eval()
    models.append(m)
print(f"Loaded! Votes/pair: {len(models)} × {args.tta} = {len(models)*args.tta}")

# ── TTA TRANSFORMS ────────────────────────────────────────────────────
_mean, _std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
tta_transforms = [
    transforms.Compose([transforms.ToTensor(), transforms.Normalize(_mean, _std)]),
    transforms.Compose([transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), transforms.Normalize(_mean, _std)]),
    transforms.Compose([transforms.ColorJitter(brightness=(1.15,1.15)), transforms.ToTensor(), transforms.Normalize(_mean, _std)]),
    transforms.Compose([transforms.ColorJitter(brightness=(0.85,0.85)), transforms.ToTensor(), transforms.Normalize(_mean, _std)]),
    transforms.Compose([transforms.RandomHorizontalFlip(p=1.0), transforms.ColorJitter(brightness=(1.1,1.1)), transforms.ToTensor(), transforms.Normalize(_mean, _std)]),
][:args.tta]

# ── IMAGE UTILS ───────────────────────────────────────────────────────
def load_img(fname):
    if not fname or str(fname).strip().lower() == 'nan': return None
    path = os.path.join(args.img_dir, str(fname).strip())
    if not os.path.exists(path):
        print(f"  WARNING: not found: {path}"); return None
    raw = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img is None: return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def to_tensors(img_np):
    pil = PILImage.fromarray(img_np)
    return [tf(pil).unsqueeze(0).to(DEVICE) for tf in tta_transforms]

# ── PREDICT ───────────────────────────────────────────────────────────
df = pd.read_csv(args.pred)
print(f"\nLoaded {len(df)} pairs. Predicting...")

winners, confidences = [], []
for i, row in df.iterrows():
    img1 = load_img(row['Image 1'])
    img2 = load_img(row['Image 2'])
    if img1 is None or img2 is None:
        winners.append(1); confidences.append(0.5); continue

    t1_list, t2_list = to_tensors(img1), to_tensors(img2)
    pair_preds = []
    with torch.no_grad():
        for t1, t2 in zip(t1_list, t2_list):
            for model in models:
                pair_preds.append(float(model(t1, t2).cpu()))

    avg_pred   = np.mean(pair_preds)
    winner     = 1 if avg_pred >= 0.5 else 2
    confidence = avg_pred if winner == 1 else (1 - avg_pred)
    winners.append(winner)
    confidences.append(round(float(confidence), 4))

    if (i + 1) % 10 == 0 or (i + 1) == len(df):
        print(f"  [{i+1}/{len(df)}]  avg={avg_pred:.3f}  "
              f"winner=Image {winner}  conf={confidence*100:.1f}%")

# ── SAVE ──────────────────────────────────────────────────────────────
df['Winner']     = winners
df['Confidence'] = confidences
df[['Image 1', 'Image 2', 'Winner']].to_csv(args.pred, index=False)
df.to_csv(args.pred.replace('.csv', '_detailed.csv'), index=False)

print(f"\n{'='*50}")
print(f"Backbone      : {args.backbone}")
print(f"Image 1 wins  : {sum(w==1 for w in winners)}")
print(f"Image 2 wins  : {sum(w==2 for w in winners)}")
print(f"Avg confidence: {np.mean(confidences)*100:.1f}%")
print(f"Votes/pair    : {len(models)*args.tta}")
print(f"{'='*50}")