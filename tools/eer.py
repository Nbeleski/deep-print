import os, random, math

from tqdm import tqdm
from PIL import Image, ImageEnhance, ImageOps
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

from itertools import combinations
from sklearn.metrics import roc_curve

def pad_to_center(img, size=448, fill=255):
    """Pads a PIL image to a square of (size, size), centering the original."""
    w, h = img.size
    pad_left = (size - w) // 2 if w < size else 0
    pad_top = (size - h) // 2 if h < size else 0
    pad_right = size - w - pad_left if w < size else 0
    pad_bottom = size - h - pad_top if h < size else 0
    img = ImageOps.expand(img, (pad_left, pad_top, pad_right, pad_bottom), fill=fill)
    if img.size[0] > size or img.size[1] > size:
        img = img.crop(((img.size[0] - size) // 2, (img.size[1] - size) // 2,
                        (img.size[0] + size) // 2, (img.size[1] + size) // 2))
    return img

def extract_features(model, device, image_paths, batch_size=4):
    """Extracts R1, R2 features for all images."""
    model.eval()
    features = []
    labels = []
    fnames = []
    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size)):
            batch_paths = image_paths[i:i+batch_size]
            imgs = []
            lbs = []
            for path in batch_paths:
                fname = os.path.basename(path)
                img = Image.open(path).convert('L')
                img = pad_to_center(img)
                img = np.array(img).astype(np.float32) / 255.0
                img = torch.from_numpy(img).unsqueeze(0)  # [1,448,448]
                imgs.append(img)
                label = fname.split('_')[0]
                lbs.append(label)
                fnames.append(fname)
            imgs = torch.stack(imgs).to(device)
            # Model output:
            out = model(imgs)
            R1 = out['R1']      # (batch, 96)
            R2 = out['R2']      # (batch, 96)
            # Concat, normalize to unit length
            emb = torch.cat([R1, R2], dim=1)
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            features.append(emb.cpu())
            labels += lbs
    features = torch.cat(features, dim=0)
    return features.numpy(), np.array(labels), fnames

def build_pairs(labels, num_impostors=10, seed=42):
    """Builds lists of genuine and impostor pairs by indices."""
    np.random.seed(seed)
    idx_by_label = {}
    for idx, lbl in enumerate(labels):
        idx_by_label.setdefault(lbl, []).append(idx)
    genuine = []
    impostor = []
    all_labels = sorted(idx_by_label.keys())
    for lbl, indices in idx_by_label.items():
        # Genuine: all combinations among indices
        if len(indices) > 1:
            genuine += list(combinations(indices, 2))
        # Impostor: for each sample, pick num_impostors random from other IDs
        others = [i for k, v in idx_by_label.items() if k != lbl for i in v]
        for i in indices:
            sampled = np.random.choice(others, min(num_impostors, len(others)), replace=False)
            for j in sampled:
                impostor.append((i, j))
    return np.array(genuine), np.array(impostor)

def cosine_similarity(a, b):
    return np.sum(a * b, axis=1)

def compute_eer(genuine_scores, impostor_scores):
    labels = np.concatenate([np.ones_like(genuine_scores), np.zeros_like(impostor_scores)])
    scores = np.concatenate([genuine_scores, impostor_scores])
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    # EER is the point where FPR == FNR
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    return eer, thresholds[eer_idx]

def evaluate_eer_on_test(model, device, test_image_dir, batch_size=32, num_impostors=10):
    image_paths = [os.path.join(test_image_dir, f)
                   for f in os.listdir(test_image_dir) if f.endswith('.png')]
    features, labels, fnames = extract_features(model, device, image_paths, batch_size)
    genuine_pairs, impostor_pairs = build_pairs(labels, num_impostors=num_impostors)
    genuine_scores = cosine_similarity(features[genuine_pairs[:, 0]], features[genuine_pairs[:, 1]])
    impostor_scores = cosine_similarity(features[impostor_pairs[:, 0]], features[impostor_pairs[:, 1]])
    eer, threshold = compute_eer(genuine_scores, impostor_scores)
    # print(f"EER: {eer:.4f} at threshold {threshold:.4f}")
    return eer, threshold