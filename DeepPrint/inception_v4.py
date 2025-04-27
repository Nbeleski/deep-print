import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Any
from torch import Tensor

class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        # super(BasicConv2d, self).__init__()
        # self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        # self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        # self.relu = nn.ReLU(True)

        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.GroupNorm(num_groups=8, num_channels=out_channels)
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
    
class ReductionA(nn.Module):
    def __init__(
            self,
            in_channels: int,
            k: int,
            l: int,
            m: int,
            n: int,
    ) -> None:
        super(ReductionA, self).__init__()
        self.branch_0 = BasicConv2d(in_channels, n, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0))
        self.branch_1 = nn.Sequential(
            BasicConv2d(in_channels, k, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            BasicConv2d(k, l, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BasicConv2d(l, m, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0)),
        )
        self.branch_2 = nn.MaxPool2d((3, 3), (2, 2))

    def forward(self, x: Tensor) -> Tensor:
        branch_0 = self.branch_0(x)
        branch_1 = self.branch_1(x)
        branch_2 = self.branch_2(x)

        out = torch.cat([branch_0, branch_1, branch_2], 1)

        return out


class InceptionB(nn.Module):
    def __init__(
            self,
            in_channels: int,
    ) -> None:
        super(InceptionB, self).__init__()
        self.branch_0 = BasicConv2d(in_channels, 384, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.branch_1 = nn.Sequential(
            BasicConv2d(in_channels, 192, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            BasicConv2d(192, 224, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3)),
            BasicConv2d(224, 256, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0)),
        )
        self.branch_2 = nn.Sequential(
            BasicConv2d(in_channels, 192, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            BasicConv2d(192, 192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0)),
            BasicConv2d(192, 224, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3)),
            BasicConv2d(224, 224, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0)),
            BasicConv2d(224, 256, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3)),
        )
        self.branch_3 = nn.Sequential(
            nn.AvgPool2d((3, 3), (1, 1), (1, 1), count_include_pad=False),
            BasicConv2d(in_channels, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
        )

    def forward(self, x: Tensor) -> Tensor:
        branch_0 = self.branch_0(x)
        branch_1 = self.branch_1(x)
        branch_2 = self.branch_2(x)
        branch_3 = self.branch_3(x)

        out = torch.cat([branch_0, branch_1, branch_2, branch_3], 1)

        return out
    
class ReductionB(nn.Module):
    def __init__(
            self,
            in_channels: int,
    ) -> None:
        super(ReductionB, self).__init__()
        self.branch_0 = nn.Sequential(
            BasicConv2d(in_channels, 192, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            BasicConv2d(192, 192, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0)),
        )
        self.branch_1 = nn.Sequential(
            BasicConv2d(in_channels, 256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            BasicConv2d(256, 256, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3)),
            BasicConv2d(256, 320, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0)),
            BasicConv2d(320, 320, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0)),
        )
        self.branch_2 = nn.MaxPool2d((3, 3), (2, 2))

    def forward(self, x: Tensor) -> Tensor:
        branch_0 = self.branch_0(x)
        branch_1 = self.branch_1(x)
        branch_2 = self.branch_2(x)

        out = torch.cat([branch_0, branch_1, branch_2], 1)

        return out

class InceptionC(nn.Module):
    def __init__(
            self,
            in_channels: int,
    ) -> None:
        super(InceptionC, self).__init__()
        self.branch_0 = BasicConv2d(in_channels, 256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

        self.branch_1 = BasicConv2d(in_channels, 384, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.branch_1_1 = BasicConv2d(384, 256, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
        self.branch_1_2 = BasicConv2d(384, 256, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))

        self.branch_2 = nn.Sequential(
            BasicConv2d(in_channels, 384, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            BasicConv2d(384, 448, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            BasicConv2d(448, 512, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
        )
        self.branch_2_1 = BasicConv2d(512, 256, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
        self.branch_2_2 = BasicConv2d(512, 256, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))

        self.branch_3 = nn.Sequential(
            nn.AvgPool2d((3, 3), (1, 1), (1, 1)),
            BasicConv2d(in_channels, 256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        )

    def forward(self, x: Tensor) -> Tensor:
        branch_0 = self.branch_0(x)
        branch_1 = self.branch_1(x)

        branch_1_1 = self.branch_1_1(branch_1)
        branch_1_2 = self.branch_1_2(branch_1)
        x1 = torch.cat([branch_1_1, branch_1_2], 1)

        branch_2 = self.branch_2(x)
        branch_2_1 = self.branch_2_1(branch_2)
        branch_2_2 = self.branch_2_2(branch_2)
        x2 = torch.cat([branch_2_1, branch_2_2], 1)

        x3 = self.branch_3(x)

        out = torch.cat([branch_0, x1, x2, x3], 1)

        return out
