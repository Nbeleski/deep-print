import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Any
from torch import Tensor

def conv_bn_relu(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU(True)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)

        return out

class InceptionV4Stem(nn.Module):
    def __init__(
            self,
            in_channels: int,
    ) -> None:
        super(InceptionV4Stem, self).__init__()
        self.conv2d_1a_3x3 = BasicConv2d(in_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0))

        self.conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        self.conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.mixed_3a_branch_0 = nn.MaxPool2d((3, 3), (2, 2))
        self.mixed_3a_branch_1 = BasicConv2d(64, 96, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0))

        self.mixed_4a_branch_0 = nn.Sequential(
            BasicConv2d(160, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            BasicConv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)),
        )
        self.mixed_4a_branch_1 = nn.Sequential(
            BasicConv2d(160, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            BasicConv2d(64, 64, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3)),
            BasicConv2d(64, 64, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0)),
            BasicConv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        )

        self.mixed_5a_branch_0 = BasicConv2d(192, 192, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0))
        self.mixed_5a_branch_1 = nn.MaxPool2d((3, 3), (2, 2))

    def forward(self, x):
        x = self.conv2d_1a_3x3(x) # 149 x 149 x 32
        x = self.conv2d_2a_3x3(x) # 147 x 147 x 32
        x = self.conv2d_2b_3x3(x) # 147 x 147 x 64
        x0 = self.mixed_3a_branch_0(x)
        x1 = self.mixed_3a_branch_1(x)
        x = torch.cat((x0, x1), dim=1) # 73 x 73 x 160
        x0 = self.mixed_4a_branch_0(x)
        x1 = self.mixed_4a_branch_1(x)
        x = torch.cat((x0, x1), dim=1) # 71 x 71 x 192
        x0 = self.mixed_5a_branch_0(x)
        x1 = self.mixed_5a_branch_1(x)
        x = torch.cat((x0, x1), dim=1) # 35 x 35 x 384
        return x

class InceptionA(nn.Module):
    def __init__(
            self,
            in_channels: int,
    ) -> None:
        super(InceptionA, self).__init__()
        self.branch_0 = BasicConv2d(in_channels, 96, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.branch_1 = nn.Sequential(
            BasicConv2d(in_channels, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            BasicConv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )
        self.branch_2 = nn.Sequential(
            BasicConv2d(in_channels, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            BasicConv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BasicConv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )
        self.brance_3 = nn.Sequential(
            nn.AvgPool2d((3, 3), (1, 1), (1, 1), count_include_pad=False),
            BasicConv2d(384, 96, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        )

    def forward(self, x):
        branch_0 = self.branch_0(x)
        branch_1 = self.branch_1(x)
        branch_2 = self.branch_2(x)
        brance_3 = self.brance_3(x)

        out = torch.cat([branch_0, branch_1, branch_2, brance_3], 1)

        return out


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
        return self.fc(self.conv(x))


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
    def __init__(self):
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

        self.minutiae_stack = nn.Sequential(*[conv_bn_relu(384, 384, 3, 1, 1) for _ in range(6)])
        self.texture_stack = nn.Sequential(
            conv_bn_relu(384, 384, 3, 1, 1),
            conv_bn_relu(384, 384, 3, 1, 1),
            conv_bn_relu(384, 384, 3, 1, 1)
        )
        self.texture_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(384, 96)
        )

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

        # TODO: T(x) should be the rest of the
        # inception V4 as far as I understand.
        t_x = self.minutiae_stack(x)
        t_feat = self.texture_stack(t_x)
        t_embed = self.texture_fc(t_feat)

        embedding = F.normalize(torch.cat([m_x, t_embed], dim=1), dim=1)

        return {
            'embedding': embedding,
            'minutiae_map': d_x,
            'alignment': align_params,
            'aligned': aligned
        }