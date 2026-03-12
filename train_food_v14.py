"""
train_food.py  —  PyTorch v14  (configurable backbone comparison)
=====================================================================
Version history:
  v11 67.2%  frozen + T=1.5 + augmented IG CSV
  v12 68.3%  patience=8, epochs=100
  v13 69.0%  hard negative mining
  v14 ???    try MobileNetV3 / ResNet50 as backbone

Motivation:
  EfficientNet-B0 was designed for ImageNet classification.
  MobileNetV3-Large uses depthwise separable convolutions →
    better at capturing local textures (broth, melt, glaze, plating)
    which is exactly what food aesthetics depends on.
  ResNet50 is a strong general baseline — deeper residual features.

How to use:
  Change BACKBONE to one of:
    'efficientnet_b0'       → 1280-dim  (current best: 69.0%)
    'mobilenetv3_large_100' → 960-dim   (try this first per paper)
    'resnet50'              → 2048-dim  (strong general baseline)
    'mobilenetv3_small_100' → 576-dim   (fastest, lightest)

  Results dir is auto-named by backbone so you can run all three
  and compare without overwriting each other.

Everything else identical to v13:
  ✅ Frozen backbone
  ✅ L2 normalize
  ✅ diff + prod head
  ✅ T=1.5, LR=2e-4, patience=8, epochs=100
  ✅ Hard negative mining (alpha=0.7, start_ep=3)
  ✅ data_from_intragram_augmented.csv
=====================================================================
"""

import os
import gc
import copy
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image as PILImage
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
import timm

torch.backends.cudnn.benchmark = True

# ======================================================================
# ★ CHANGE THIS TO SWITCH BACKBONE ★
# ======================================================================
BACKBONE = 'efficientnet_b3'
# Options:
#   'efficientnet_b0'        1280-dim  v13 baseline 69.0%
#   'efficientnet_b3'        1536-dim  ★ bigger EfficientNet, more texture detail
#   'mobilenetv3_large_100'  960-dim   paper recommendation
#   'resnet50'               2048-dim  deep residual baseline
#   'mobilenetv3_small_100'  576-dim   fastest/lightest
# ======================================================================

# ======================================================================
# CONFIG
# ======================================================================
Q_IMG_PATH    = 'Questionair Images/'
Q_CSV         = 'data_from_questionaire.csv'
IG_IMG_ROOT   = 'Intragram Images [Original]/'
IG_CSV        = 'data_from_intragram_augmented.csv'

# Auto-name result dir by backbone so runs don't overwrite each other
BACKBONE_SHORT = BACKBONE.replace('_', '-')
RESULT_DIR     = f'result/food_model_{BACKBONE_SHORT}'

CATEGORIES         = ['Burger', 'Dessert', 'Pizza', 'Ramen', 'Sushi']
IMG_SIZE           = 224
BATCH_SIZE         = 64
EPOCHS             = 100
N_FOLDS            = 5
Q_WEIGHT           = 1.0
IG_WEIGHT          = 1.5
TEMPERATURE        = 1.5
LR                 = 2e-4
PATIENCE           = 8
MINING_START_EPOCH = 3
MINING_ALPHA       = 0.7

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device  : {DEVICE}")
print(f"GPU     : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"Backbone: {BACKBONE}")

os.makedirs(RESULT_DIR, exist_ok=True)

# ======================================================================
# TRANSFORMS
# ======================================================================
aug_tf = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
base_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ======================================================================
# PRELOAD IMAGES INTO RAM
# ======================================================================
def preload_images(paths):
    cache, failed = {}, 0
    for path in tqdm(paths, desc="  Preloading", ncols=80, leave=False):
        if path in cache:
            continue
        try:
            raw = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            if img is None: raise ValueError()
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cache[path] = base_tf(PILImage.fromarray(img))
        except Exception:
            cache[path] = torch.zeros(3, IMG_SIZE, IMG_SIZE)
            failed += 1
    if failed:
        print(f"  WARNING: {failed} images failed (zeros used)")
    return cache

# ======================================================================
# DATASET
# ======================================================================
class PairDataset(Dataset):
    def __init__(self, paths1, paths2, labels, weights, cache, augment=False):
        self.paths1  = paths1
        self.paths2  = paths2
        self.labels  = torch.tensor(labels,  dtype=torch.float32)
        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.cache   = cache
        self.augment = augment

    def __len__(self): return len(self.labels)

    def _get(self, path):
        t = self.cache[path]
        if self.augment:
            img = PILImage.fromarray(
                (t.permute(1,2,0).numpy()*255).clip(0,255).astype(np.uint8))
            return aug_tf(img)
        return t

    def __getitem__(self, idx):
        return (self._get(self.paths1[idx]),
                self._get(self.paths2[idx]),
                self.labels[idx],
                self.weights[idx],
                idx)

# ======================================================================
# LOAD PAIRS
# ======================================================================
def load_all_pairs():
    df_q = pd.read_csv(Q_CSV)
    q_pairs, skipped = [], 0
    for _, row in df_q.iterrows():
        n1 = str(row["Image 1"]).strip()
        n2 = str(row["Image 2"]).strip()
        if not n1.lower().endswith((".jpg",".jpeg",".png")): n1 += ".jpg"
        if not n2.lower().endswith((".jpg",".jpeg",".png")): n2 += ".jpg"
        p1 = os.path.join(Q_IMG_PATH, n1)
        p2 = os.path.join(Q_IMG_PATH, n2)
        if not os.path.exists(p1) or not os.path.exists(p2):
            skipped += 1; continue
        label = 1.0 if int(row["Winner"]) == 1 else 0.0
        q_pairs.append((p1, p2, label, Q_WEIGHT))
    print(f"  [questionnaire] {len(q_pairs)} pairs  (skipped {skipped})")

    df_ig = pd.read_csv(IG_CSV)
    ig_pairs, skipped = [], 0
    for _, row in df_ig.iterrows():
        menu = str(row["Menu"]).strip()
        if menu not in CATEGORIES: skipped += 1; continue
        n1 = str(row["Image 1"]).strip()
        n2 = str(row["Image 2"]).strip()
        if not n1.lower().endswith((".jpg",".jpeg",".png")): n1 += ".jpg"
        if not n2.lower().endswith((".jpg",".jpeg",".png")): n2 += ".jpg"
        p1 = os.path.join(IG_IMG_ROOT, menu, n1)
        p2 = os.path.join(IG_IMG_ROOT, menu, n2)
        if not os.path.exists(p1) or not os.path.exists(p2):
            skipped += 1; continue
        label = 1.0 if int(row["Winner"]) == 1 else 0.0
        ig_pairs.append((p1, p2, label, IG_WEIGHT))
    print(f"  [instagram]     {len(ig_pairs)} pairs  (skipped {skipped})")
    print(f"  [total]         {len(q_pairs)+len(ig_pairs)} pairs")
    return q_pairs + ig_pairs

# ======================================================================
# HARD MINING
# ======================================================================
@torch.no_grad()
def compute_mining_weights(model, dataset, alpha=MINING_ALPHA):
    model.eval()
    criterion = nn.BCELoss(reduction='none')
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
    all_losses = []
    for batch in loader:
        x1_b, x2_b, y_b = batch[0].to(DEVICE), batch[1].to(DEVICE), batch[2].to(DEVICE)
        pred = model(x1_b, x2_b)
        loss = criterion(pred.float(), y_b.float())
        all_losses.append(loss.cpu())
    losses = torch.cat(all_losses)
    loss_weights = losses / (losses.sum() + 1e-8)
    uniform = torch.ones_like(loss_weights) / len(loss_weights)
    final_weights = alpha * loss_weights + (1 - alpha) * uniform
    final_weights = final_weights / final_weights.sum()
    return final_weights.numpy(), losses.mean().item()

# ======================================================================
# MODEL v14 — backbone is now configurable
#
# Feature dimensions by backbone:
#   efficientnet_b0        → feat=1280  cmp=2560
#   mobilenetv3_large_100  → feat=960   cmp=1920
#   resnet50               → feat=2048  cmp=4096
#   mobilenetv3_small_100  → feat=576   cmp=1152
#
# Head size scales automatically with backbone output dim.
# ======================================================================
class SiameseNet(nn.Module):
    def __init__(self, backbone_name=BACKBONE):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=True, num_classes=0
        )
        for param in self.backbone.parameters():
            param.requires_grad = False

        feat_dim = self.backbone.num_features
        cmp_dim  = feat_dim * 2   # diff + prod concatenated

        # Head size scales with backbone — keep ratio consistent
        hidden = max(128, cmp_dim // 10)   # ~256 for mobilenet, ~400 for resnet

        self.comparison_head = nn.Sequential(
            nn.Linear(cmp_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.45),
            nn.Linear(hidden, 1),
        )
        self.temperature = TEMPERATURE

        print(f"  Backbone      : {backbone_name}")
        print(f"  Feature dim   : {feat_dim}")
        print(f"  Comparison dim: {cmp_dim}  (diff + prod)")
        print(f"  Head hidden   : {hidden}")
        n_head = sum(p.numel() for p in self.comparison_head.parameters())
        print(f"  Head params   : {n_head:,}")

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
        logit = self.comparison_head(combined).squeeze(1)
        return torch.sigmoid(logit / self.temperature)

# ======================================================================
# EPOCH RUNNER
# ======================================================================
def run_epoch(model, loader, optimizer=None, scaler=None):
    is_train  = optimizer is not None
    model.train() if is_train else model.eval()
    criterion = nn.BCELoss(reduction='none')
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader,
                desc="  train" if is_train else "  val  ",
                ncols=90, leave=False)

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in pbar:
            x1_b, x2_b, y_b, w_b = batch[0], batch[1], batch[2], batch[3]
            x1_b, x2_b = x1_b.to(DEVICE), x2_b.to(DEVICE)
            y_b,  w_b  = y_b.to(DEVICE),  w_b.to(DEVICE)

            with autocast('cuda'):
                pred = model(x1_b, x2_b)
            loss = (criterion(pred.float(), y_b.float()) * w_b).mean()

            if is_train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * x1_b.size(0)
            correct    += ((pred > 0.5) == y_b.bool()).sum().item()
            total      += x1_b.size(0)
            pbar.set_postfix(loss=f"{total_loss/total:.4f}",
                             acc=f"{correct/total*100:.1f}%")

    return total_loss / total, correct / total

# ======================================================================
# MAIN
# ======================================================================
if __name__ == '__main__':

    print("="*60)
    print(f"v14 — backbone={BACKBONE}")
    print(f"T={TEMPERATURE}  |  LR={LR}  |  patience={PATIENCE}")
    print(f"Mining alpha={MINING_ALPHA}, start_ep={MINING_START_EPOCH}")
    print(f"Results → {RESULT_DIR}")
    print("="*60)

    all_pairs = load_all_pairs()
    P1 = [r[0] for r in all_pairs]
    P2 = [r[1] for r in all_pairs]
    Y  = [r[2] for r in all_pairs]
    W  = [r[3] for r in all_pairs]

    all_paths = list(set(P1 + P2))
    print(f"\nPreloading {len(all_paths)} unique images into RAM...")
    cache = preload_images(all_paths)
    print(f"Done — {len(cache)} images cached\n")

    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_val_accs, all_histories, best_epoch_idxs = [], [], []

    print("="*60)
    print(f"Training {N_FOLDS}-fold CV")
    print("="*60)

    for fold, (tr_idx, va_idx) in enumerate(
            kf.split(np.zeros(len(Y)), np.array(Y))):

        print(f"\n{'='*50}  FOLD {fold+1}/{N_FOLDS}  {'='*50}")

        p1_tr = [P1[i] for i in tr_idx] + [P2[i] for i in tr_idx]
        p2_tr = [P2[i] for i in tr_idx] + [P1[i] for i in tr_idx]
        y_tr  = [Y[i]  for i in tr_idx] + [1.0-Y[i] for i in tr_idx]
        w_tr  = [W[i]  for i in tr_idx] + [W[i]     for i in tr_idx]
        p1_va = [P1[i] for i in va_idx]
        p2_va = [P2[i] for i in va_idx]
        y_va  = [Y[i]  for i in va_idx]
        w_va  = [1.0]  * len(y_va)

        print(f"  Train: {len(tr_idx)} → {len(p1_tr)} (swap aug)  |"
              f"  Val: {len(p1_va)}")

        train_ds = PairDataset(p1_tr, p2_tr, y_tr, w_tr, cache, augment=True)
        val_ds   = PairDataset(p1_va, p2_va, y_va, w_va, cache, augment=False)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                                shuffle=False, num_workers=0)

        model     = SiameseNet().to(DEVICE)
        optimizer = optim.Adam(model.comparison_head.parameters(),
                               lr=LR, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS, eta_min=1e-6
        )
        scaler = GradScaler('cuda')

        best_loss, best_acc = float('inf'), 0.0
        best_epoch, best_state = 0, None
        no_improve = 0
        history = {'val_acc': [], 'val_loss': []}
        sample_weights = np.ones(len(train_ds)) / len(train_ds)

        for epoch in range(EPOCHS):
            if epoch >= MINING_START_EPOCH:
                sampler = WeightedRandomSampler(
                    weights=torch.from_numpy(sample_weights).float(),
                    num_samples=len(train_ds), replacement=True)
                train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                                          sampler=sampler, num_workers=0)
            else:
                train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=0)

            tr_loss, tr_acc = run_epoch(model, train_loader, optimizer, scaler)
            va_loss, va_acc = run_epoch(model, val_loader)
            scheduler.step()

            history['val_acc'].append(va_acc)
            history['val_loss'].append(va_loss)

            if epoch >= MINING_START_EPOCH - 1:
                sample_weights, avg_hard_loss = \
                    compute_mining_weights(model, train_ds)
                mining_tag = f"  [hard:{avg_hard_loss:.3f}]"
            else:
                mining_tag = "  [warmup]"

            improved = va_loss < best_loss
            print(f"  Ep {epoch+1:02d}/{EPOCHS}  "
                  f"tr={tr_loss:.4f}/{tr_acc*100:.1f}%  "
                  f"val={va_loss:.4f}/{va_acc*100:.1f}%  "
                  f"lr={optimizer.param_groups[0]['lr']:.1e}"
                  f"{mining_tag}"
                  + ("  ★" if improved else ""))

            if improved:
                best_loss, best_acc = va_loss, va_acc
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                torch.save(best_state,
                           f"{RESULT_DIR}/fold{fold+1}_best.pth")
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= PATIENCE:
                    print(f"  Early stop @ epoch {epoch+1}  "
                          f"(best epoch {best_epoch+1})")
                    break

        model.load_state_dict(best_state)
        fold_val_accs.append(best_acc)
        all_histories.append(history)
        best_epoch_idxs.append(best_epoch)
        print(f"  Fold {fold+1} → best val acc: {best_acc*100:.2f}%  "
              f"@ epoch {best_epoch+1}")

        del model, optimizer, scaler
        torch.cuda.empty_cache(); gc.collect()

    print(f"\n{'='*55}")
    print(f"CROSS-VALIDATION SUMMARY  [{BACKBONE}]")
    for i, acc in enumerate(fold_val_accs):
        print(f"  Fold {i+1}: {acc*100:.2f}%")
    print(f"  Mean : {np.mean(fold_val_accs)*100:.2f}%  "
          f"±  {np.std(fold_val_accs)*100:.2f}%")
    print(f"{'='*55}")

    avg_best_epoch = int(np.mean(best_epoch_idxs)) + 1
    print(f"\nFinal model: {avg_best_epoch} epochs on all data...")

    p1_all = P1+P2;  p2_all = P2+P1
    y_all  = Y + [1.0-v for v in Y]
    w_all  = W + W

    all_ds     = PairDataset(p1_all, p2_all, y_all, w_all, cache, augment=True)
    all_loader = DataLoader(all_ds, batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=0)

    final     = SiameseNet().to(DEVICE)
    optimizer = optim.Adam(final.comparison_head.parameters(),
                           lr=LR, weight_decay=1e-4)
    scaler    = GradScaler('cuda')

    for epoch in range(avg_best_epoch):
        loss, acc = run_epoch(final, all_loader, optimizer, scaler)
        print(f"  Ep {epoch+1:02d}/{avg_best_epoch}  "
              f"loss={loss:.4f}  acc={acc*100:.1f}%")

    torch.save(final.state_dict(), f"{RESULT_DIR}/final_model.pth")
    print(f"Final model saved → {RESULT_DIR}/final_model.pth")

    del final, optimizer, scaler, all_loader, all_ds
    torch.cuda.empty_cache()

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, N_FOLDS))
    for fold, h in enumerate(all_histories):
        axes[0].plot(h['val_acc'],  color=colors[fold],
                     label=f"Fold {fold+1}  ({fold_val_accs[fold]*100:.1f}%)")
        axes[1].plot(h['val_loss'], color=colors[fold],
                     label=f"Fold {fold+1}")
    axes[0].axhline(np.mean(fold_val_accs), color='black', linestyle='--',
                    label=f"Mean {np.mean(fold_val_accs)*100:.1f}%")
    axes[0].set_title('Validation Accuracy per Fold')
    axes[0].set_xlabel('Epoch'); axes[0].legend(fontsize=8); axes[0].grid(True)
    axes[1].set_title('Validation Loss per Fold')
    axes[1].set_xlabel('Epoch'); axes[1].legend(fontsize=8); axes[1].grid(True)
    plt.suptitle(
        f'Siamese Food Ranker v14 — {BACKBONE}  '
        f'(T={TEMPERATURE}, mining α={MINING_ALPHA})',
        fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/learning_curve.png", dpi=150)
    print(f"Curve saved → {RESULT_DIR}/learning_curve.png")
    print("\nDone!")