import os, random, math

from PIL import Image, ImageEnhance
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

def apply_affine_to_img(img_np, dx, dy, dt):
    # Convert to torch tensor and add batch/channel dims
    img_tensor = torch.from_numpy(img_np).float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    H, W = img_tensor.shape[-2:]

    tx = dx / W
    ty = dy / H
    theta = dt

    cos_theta = torch.cos(torch.tensor(theta))
    sin_theta = torch.sin(torch.tensor(theta))

    # Affine matrix: [1, 2, 3]
    affine = torch.tensor([[[cos_theta, -sin_theta, tx],
                            [sin_theta,  cos_theta, ty]]], dtype=torch.float)

    grid = F.affine_grid(affine, img_tensor.size(), align_corners=False)
    out = F.grid_sample(img_tensor, grid, align_corners=False, padding_mode='border')
    return out.squeeze().numpy()  # Remove batch/channel dims and return NumPy

def rotate_orientation_channels(mmap_rotated, delta_theta):
    """
    Rotates orientation bins of a 6-channel map by delta_theta radians.
    mmap_rotated: (H, W, 6) — already spatially rotated
    """
    H, W, C = mmap_rotated.shape
    assert C == 6

    # Compute relative circular shift
    bin_shift = -6 * delta_theta / (2 * np.pi)  # negative = CCW rotation

    # Circular interpolation along channel axis
    fft = np.fft.fft(mmap_rotated, axis=2)
    freqs = np.fft.fftfreq(C)[:, None, None]  # shape (6, 1, 1)
    phase = np.exp(2j * np.pi * freqs * bin_shift)
    fft_shifted = fft * phase.T  # transpose for broadcasting
    mmap_rotated_oriented = np.fft.ifft(fft_shifted, axis=2).real

    return mmap_rotated_oriented

def apply_affine_to_map(mmap_np, dx, dy, dt, img_shape=(448, 448)):
    H, W, C = mmap_np.shape
    assert C == 6, "Expected 6 channels in minutiae map"

    # Convert to [1, 6, H, W] tensor
    mmap_tensor = torch.from_numpy(mmap_np).float().permute(2, 0, 1).unsqueeze(0)  # [1, 6, H, W]

    ih, iw = img_shape
    tx = dx / iw
    ty = dy / ih    
    theta = dt

    cos_theta = torch.cos(torch.tensor(theta))
    sin_theta = torch.sin(torch.tensor(theta))

    affine = torch.tensor([[[cos_theta, -sin_theta, tx],
                            [sin_theta,  cos_theta, ty]]], dtype=torch.float)
    # affine = torch.tensor([[[1.0, 0.0, tx],
    #                         [0.0, 1.0, ty]]], dtype=torch.float)

    grid = F.affine_grid(affine, mmap_tensor.size(), align_corners=False)
    transformed = F.grid_sample(mmap_tensor, grid, mode='nearest', align_corners=False)

    # Convert back to [H, W, 6]
    # return transformed.squeeze(0).permute(1, 2, 0).numpy()
    return rotate_orientation_channels(
        transformed.squeeze(0).permute(1, 2, 0).numpy(), dt
    )

def rotate_orientation_channels_torch(mmap_tensor, delta_theta):
    """
    Rotates the orientation channels of a 6-channel minutiae map tensor (circular shift).

    Args:
        mmap_tensor: torch.Tensor of shape [1, 6, H, W] (or [B, 6, H, W])
        delta_theta: scalar float in radians (clockwise positive)

    Returns:
        Rotated tensor, same shape [B, 6, H, W]
    """
    B, C, H, W = mmap_tensor.shape
    assert C == 6, "Expected 6 orientation bins"

    # Compute fractional shift
    shift = -6 * delta_theta / (2 * torch.pi)  # counter-clockwise

    # DFT over orientation dimension (dim=1)
    fmap = torch.fft.fft(mmap_tensor, dim=1)

    # Frequency indices (shape [C], from 0 to C-1)
    freqs = torch.fft.fftfreq(C, device=mmap_tensor.device).view(1, C, 1, 1)

    # Complex exponential phase shift
    phase = torch.exp(-2j * torch.pi * freqs * shift)

    # Apply phase shift (broadcast over B, H, W)
    fmap_shifted = fmap * phase

    # Inverse FFT
    rotated = torch.fft.ifft(fmap_shifted, dim=1).real

    return rotated

def apply_affine_to_map_tensor(mmap_tensor, dx, dy, dt, img_shape=(448, 448)):
    """
    Applies affine transform (translation + rotation) to a 6-channel minutiae map tensor.

    Args:
        mmap_tensor: torch.Tensor of shape [1, 6, H, W]
        dx, dy: translation in image pixels (same scale as img_shape)
        dt: rotation in radians
        img_shape: shape of source image that the map aligns to (used to normalize dx, dy)

    Returns:
        Transformed map tensor, same shape [1, 6, H, W]
    """
    assert mmap_tensor.dim() == 4 and mmap_tensor.size(1) == 6, "Expected shape [1, 6, H, W]"
    device = mmap_tensor.device
    _, _, H, W = mmap_tensor.shape
    ih, iw = img_shape

    tx = dx / iw
    ty = dy / ih

    theta = torch.tensor(dt, device=device)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    affine = torch.tensor([[[cos_theta, -sin_theta, tx],
                            [sin_theta,  cos_theta, ty]]], dtype=torch.float, device=device)

    grid = F.affine_grid(affine, mmap_tensor.size(), align_corners=False)
    transformed = F.grid_sample(mmap_tensor, grid, mode='nearest', align_corners=False)

    # Rotate orientation channels
    rotated = rotate_orientation_channels_torch(transformed, dt)
    #rotated_tensor = rotated.permute(2, 0, 1).unsqueeze(0).to(device)
    return rotated

class DeepPrintDataset(Dataset):
    def __init__(self, root, split='train', ids=None, color_jitter=True, affine_aug=True, id2label=None):
        """
        root: path to dataset root (expects images/ and maps/ subdirs)
        split: 'train' or 'val'
        ids: list of (id, sample_idx) tuples to use, or None to scan all
        color_jitter: whether to use brightness/contrast/sharpness augmentation (train only)
        affine_aug: whether to use random affine (rotation/translation) augmentation (train only)
        """
        self.img_dir = os.path.join(root, "images")
        self.map_dir = os.path.join(root, "maps")
        self.split = split
        self.color_jitter = color_jitter if split == 'train' else False
        self.affine_aug = affine_aug if split == 'train' else False

        if ids is None:
            self.samples = sorted([
                fname[:-4] for fname in os.listdir(self.img_dir)
                if fname.endswith('.png') and os.path.isfile(os.path.join(self.map_dir, fname.replace('.png', '.npy')))
            ])
        else:
            self.samples = ids

        if id2label is not None:
            self.id2label = id2label
        else:
            # Fallback to old behavior if no mapping is provided
            self.id2label = {}
            for s in self.samples:
                id_str = s.split('_')[0]
                if id_str not in self.id2label:
                    self.id2label[id_str] = len(self.id2label)
        self.num_classes = len(self.id2label)
        print(f'{len(self.samples)} images for {len(self.id2label)} classes found for {self.split} split')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname = self.samples[idx]
        img_path = os.path.join(self.img_dir, fname + '.png')
        map_path = os.path.join(self.map_dir, fname + '.npy')
        img = Image.open(img_path).convert('L')
        h_img, w_img = img.height, img.width

        H = np.load(map_path)
        H = torch.from_numpy(H.transpose(2,0,1)).float()
        h_map, w_map = H.shape[1], H.shape[2]

        # Optional: augment image color
        if self.color_jitter:
            enhancers = [
                ImageEnhance.Contrast,
                ImageEnhance.Brightness,
                ImageEnhance.Sharpness
            ]
            for enhancer in enhancers:
                factor = random.uniform(0.8, 1.2)
                img = enhancer(img).enhance(factor)

        # Optional: joint affine (rotation, translation) - affects both image and map
        # map has to be interpolated to new values, complicated stuff lol.
        if self.affine_aug:
            angle = random.uniform(-15, 15)
            tx = random.uniform(-0.05, 0.05) * w_img
            ty = random.uniform(-0.05, 0.05) * h_img

            # Convert PIL image to numpy
            img_np = np.array(img, dtype=np.float32)
            img_np = img_np / 255.0  # normalize

            # Apply joint affine
            img_np = apply_affine_to_img(img_np, dx=tx, dy=ty, dt=np.radians(angle))
            mmap_np = H.permute(1, 2, 0).numpy()
            mmap_np = apply_affine_to_map(mmap_np, dx=tx, dy=ty, dt=np.radians(angle), img_shape=(h_img, w_img))

            # Convert back to tensor objects
            img = TF.to_pil_image(np.clip(img_np * 255, 0, 255).astype(np.uint8))
            H = torch.from_numpy(mmap_np).permute(2, 0, 1).float()

        # Random crop
        crop_size = 448
        map_crop = 192

        # TODO: we may want to accept smaller images later...
        assert h_img >= crop_size and w_img >= crop_size, "Image too small!"
        assert h_map >= map_crop and w_map >= map_crop, "Map too small!" 

        x_img = random.randint(0, w_img - crop_size)
        y_img = random.randint(0, h_img - crop_size)
        x_map = int(x_img * (w_map / w_img))
        y_map = int(y_img * (h_map / h_img))
        img = img.crop((x_img, y_img, x_img + crop_size, y_img + crop_size))
        H = H[:, y_map:y_map+map_crop, x_map:x_map+map_crop]
        # H *= 100.

        # To tensor, normalize to [0,1]
        img = TF.to_tensor(img).float()
        label = self.id2label[fname.split('_')[0]]
        return img, label, H