import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from typing import Any
from torch import Tensor

from .inception_v4 import InceptionV4Stem, InceptionA
from .inception_v4 import InceptionB, InceptionC, ReductionA, ReductionB

def conv_bn_relu(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
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

    def forward(self, x):
        x = F.interpolate(x, size=(128, 128), mode='bilinear')
        # return self.fc(self.conv(x))
        params = self.fc(self.conv(x))
        params[:, 0:2] = torch.clamp(params[:, 0:2], -224, 224)   # translation
        params[:, 2] = torch.clamp(params[:, 2], -np.pi/3, np.pi/3)  # rotation
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
            conv_bn_relu(384, 128, 3, 1, 1),  # 70×70
            nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1),            # 70 → 140
            conv_bn_relu(128, 64, 3, 1, 1),   # 140×140
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=0),              # 140 → 281 (crop later)
            conv_bn_relu(32, 32, 3, 1, 1),
            nn.Conv2d(32, 6, 1)               # Final 6 channels
        )

    def forward(self, x):
        x = self.net(x)
        # center crop to 192x192
        H, W = x.shape[-2:]
        top = (H - 192) // 2
        left = (W - 192) // 2
        return x[..., top:top+192, left:left+192]



class DeepPrintNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.localization = LocalizationNetwork()
        self.sampler = GridSampler()
        self.stem = InceptionV4Stem(in_channels=1)

        self.inceptionA = nn.Sequential(
            InceptionA(384),
            InceptionA(384),
            InceptionA(384),
            InceptionA(384),
            # InceptionA(384),
            # InceptionA(384),
        )

        self.minutiae_embed = MinutiaeEmbeddingHead(384)
        self.minutiae_map = MinutiaeMapHead(384)

        # T(x), but technically the rest of
        # the inceptionV4
        k: int = 192
        l: int = 224
        m: int = 256
        n: int = 384
        self.T = nn.Sequential(
            ReductionA(384, k, l, m, n),
            InceptionB(1024),
            InceptionB(1024),
            InceptionB(1024),
            InceptionB(1024),
            InceptionB(1024),
            InceptionB(1024),
            InceptionB(1024),
            ReductionB(1024),
            InceptionC(1536),
            InceptionC(1536),
            InceptionC(1536),
        )

        self.texture_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(1536, 96)
        )

        # logits for training only:
        self.classifier1 = nn.Linear(96, num_classes)
        self.classifier2 = nn.Linear(96, num_classes)

    def forward(self, x):
        align_params = self.localization(x)
        aligned = self.sampler(x, align_params)

        x = self.stem(aligned)

        # as far as I understand
        # E(x) is running the input through
        # Inception A block 6x times, although
        # the official InceptionV4 is 4x only.
        e_x = self.inceptionA(x)
        d_x = self.minutiae_map(e_x)
        m_x = self.minutiae_embed(e_x)

        # Then the T(x) is the rest of the
        # inception V4
        t_x = self.T(x)
        t_x = self.texture_fc(t_x)

        embedding = F.normalize(torch.cat([m_x, t_x], dim=1), dim=1)

        logits1 = self.classifier1(m_x)   # [batch, num_classes]
        logits2 = self.classifier2(t_x)   # [batch, num_classes]

        # We will be outputting both R1, R2 and
        # the logits for each, simply because
        # these values will be used on the loss 
        # calculation for training.
        return {
            'embedding': embedding,
            'minutiae_map': d_x,
            'alignment': align_params,
            'aligned': aligned,
            'R1': m_x,
            'R2': t_x,
            'logits_r1': logits1,
            'logits_r2': logits2
        }