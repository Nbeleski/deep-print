import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

import numpy as np
import copy
from typing import Any
from torch import Tensor

# Your heads stay the same as before
from .inception_v4 import InceptionA  # If you want to keep your version for the Minutiae branch

def conv_bn_relu(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return nn.Sequential(
        # nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        # nn.BatchNorm2d(out_channels),
        # nn.ReLU(inplace=True)

        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.GroupNorm(num_groups=8, num_channels=out_channels),
        nn.ReLU(inplace=True)
    )

class LocalizationNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 24, 5), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(24, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 48, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(48, 64, 3), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*6*6, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

        # ---- Zero the final FC layer's bias (and optionally weights) ----
        # self.fc[-1] is the nn.Linear(64, 3)
        nn.init.zeros_(self.fc[-1].weight)
        nn.init.zeros_(self.fc[-1].bias)

    def forward(self, x):
        x = F.interpolate(x, size=(128, 128), mode='bilinear')
        params = self.fc(self.conv(x))
        # Clamp everything out-of-place
        # Split into translation and rotation, clamp separately, then concatenate
        trans = torch.clamp(params[:, 0:2], -224, 224)
        rot = torch.clamp(params[:, 2:3], -np.pi/4, np.pi/4)  # keep dimension
        params = torch.cat([trans, rot], dim=1)
        return params


class GridSampler(nn.Module):
    def forward(self, img, params):
        B, _, H, W = img.shape
        tx = params[:, 0] / W
        ty = params[:, 1] / H
        theta = params[:, 2]

        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        affine = torch.zeros(B, 2, 3, device=img.device)
        affine[:, 0, 0] = cos_theta
        affine[:, 0, 1] = -sin_theta
        affine[:, 1, 0] = sin_theta
        affine[:, 1, 1] = cos_theta
        affine[:, :, 2] = torch.stack([tx, ty], dim=1)

        grid = F.affine_grid(affine, img.size(), align_corners=False)
        return F.grid_sample(img, grid, align_corners=False)


class MinutiaeEmbeddingHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.net = nn.Sequential(
            conv_bn_relu(in_channels, 384, 3, 1, 1),
            conv_bn_relu(384, 768, 3, 2, 1),
            conv_bn_relu(768, 768, 3, 1, 1),
            conv_bn_relu(768, 896, 3, 2, 1),
            conv_bn_relu(896, 1024, 3, 1, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, 96)
        )

    def forward(self, x):
        return self.net(x)


class MinutiaeMapHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 384, 3, stride=2, padding=1, output_padding=1),   # 35 → 70
            conv_bn_relu(384, 128, 7, 1, 1),  # 70×70
            nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1),            # 70 → 140
            conv_bn_relu(128, 32, 3, 1, 1),   # 140×140
            nn.Conv2d(32, 6, 1)  
        )

    def forward(self, x):
        x = self.net(x)
        start = (x.shape[-2] - 192) // 2
        end = start + 192
        x = x[..., start:end, start:end]
        return x

def modify_first_conv_to_1ch(model):
    # The first conv layer is model.features[0].conv
    old_conv = model.features[0].conv
    old_weights = old_conv.weight.data  # shape [out_c, in_c=3, k, k]
    # Average across the input channel (dim=1) to get [out_c, 1, k, k]
    new_weights = old_weights.mean(dim=1, keepdim=True)
    # Create a new Conv2d with 1 input channel
    new_conv = nn.Conv2d(1, old_conv.out_channels, old_conv.kernel_size, old_conv.stride, old_conv.padding, bias=False)
    new_conv.weight.data = new_weights
    model.features[0].conv = new_conv
    return model

# features[0]: ConvNormAct (conv2d_1a)
# features[1]: ConvNormAct (conv2d_2a)
# features[2]: ConvNormAct (conv2d_2b)
# features[3]: Mixed_3a
# features[4]: Mixed_4a
# features[5]: Inception_A 1
# features[6]: Inception_A 2
# features[7]: Inception_A 3
# features[8]: Inception_A 4

class DeepPrintNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.localization = LocalizationNetwork()
        self.sampler = GridSampler()

        # === Pre-trained Inception-v4 backbone from timm ===
        base = timm.create_model('inception_v4', pretrained=True, num_classes=0)
        base = modify_first_conv_to_1ch(base)

        # === Shared Stem ===
        # features[0]..[4]: conv, conv, conv, Mixed_3a, Mixed_4a
        self.stem = nn.Sequential(*[base.features[i] for i in range(5)])  # [0,1,2,3,4]

        # === Minutiae branch (Inception-A blocks) ===
        # we are making a deep copy here, each side of the model has its own Inception A blocks.
        self.minutiae_inceptionA = nn.Sequential(
            copy.deepcopy(base.features[5]),
            copy.deepcopy(base.features[6]),
            copy.deepcopy(base.features[7]),
            copy.deepcopy(base.features[8])
        )
        # For true DeepPrint, they show 6 blocks though.

        # Your custom heads
        self.minutiae_embed = MinutiaeEmbeddingHead(384)    # Inception-A output is 384 channels
        self.minutiae_map = MinutiaeMapHead(384)            # "

        # === Texture branch ===
        # features[9]: Reduction-A
        # features[10]..[16]: Inception-B (7 blocks)
        # features[17]: Reduction-B
        # features[18..20]: Inception-C (3 blocks)
        self.texture_blocks = nn.Sequential(*[base.features[i] for i in range(5, 21)])  # 9 to 20 inclusive
        self.texture_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(1536, 96)  # Inception-C output is 1536 channels
        )

        # Classifiers (for training/loss only)
        self.classifier1 = nn.Linear(96, num_classes)
        self.classifier2 = nn.Linear(96, num_classes)

    def forward(self, x):
        # --- Alignment ---
        align_params = self.localization(x)
        aligned = self.sampler(x, align_params)

        # --- Shared Stem ---
        x_stem = self.stem(aligned)       # (batch, 192, 71, 71)

        # --- Minutiae Branch ---
        e_x = self.minutiae_inceptionA(x_stem)  # (batch, 384, 71, 71)
        d_x = self.minutiae_map(e_x)            # (batch, 6, H, W) as you define
        m_x = self.minutiae_embed(e_x)          # (batch, 96)

        # --- Texture Branch ---
        t_x = self.texture_blocks(x_stem)       # (batch, 1536, H', W')
        t_x = self.texture_fc(t_x)              # (batch, 96)

        # --- Embedding: concatenate and normalize ---
        embedding = F.normalize(torch.cat([m_x, t_x], dim=1), dim=1)  # (batch, 192)

        logits1 = self.classifier1(m_x)
        logits2 = self.classifier2(t_x)

        return {
            'embedding': embedding,
            'minutiae_map': d_x,
            'alignment': align_params,
            'aligned': aligned,
            'R1': m_x,
            'R2': t_x,
            'logits1': logits1,
            'logits2': logits2
        }