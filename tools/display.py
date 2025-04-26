import matplotlib.pyplot as plt
import numpy as np
import cv2
import math, os
from scipy.ndimage import maximum_filter

def extract_minutiae_from_map(H, threshold=0.5, nms_dist=8):
    """
    Convert 6-channel minutiae map H to list of (x, y, theta), using local maxima and non-max suppression,
    with sub-channel (quadratic) orientation refinement.
    Assumes orientation bins are in [0, 2pi).
    Args:
        H: (H, W, 6) numpy array, channels represent evenly spaced orientations in [0, 2pi)
        threshold: minimum value for candidate minutia
        nms_dist: non-max suppression radius (pixels)
    Returns:
        minutiae: list of (x, y, theta)
    """
    h, w, n_ori = H.shape
    candidates = []

    bin_width = 2 * np.pi / n_ori  # now 2pi for 6 channels

    for k in range(n_ori):
        channel = H[..., k]
        maxima = (channel == maximum_filter(channel, size=5))
        peaks = np.argwhere(maxima & (channel > threshold))
        for y, x in peaks:
            y_m1 = H[y, x, (k - 1) % n_ori]
            y_0  = H[y, x, k]
            y_p1 = H[y, x, (k + 1) % n_ori]
            denom = y_m1 - 2 * y_0 + y_p1
            if denom == 0:
                delta = 0
            else:
                delta = 0.5 * (y_m1 - y_p1) / denom

            theta = ((k + delta) * bin_width) % (2 * np.pi)
            score = y_0
            candidates.append((x, y, theta, score))

    # Non-maximal suppression across all orientations
    if not candidates:
        return []
    candidates.sort(key=lambda tup: -tup[3])
    selected = []
    taken = np.zeros((h, w), dtype=bool)
    for x, y, theta, score in candidates:
        if taken[y, x]:
            continue
        y0 = max(0, y - nms_dist)
        y1 = min(h, y + nms_dist + 1)
        x0 = max(0, x - nms_dist)
        x1 = min(w, x + nms_dist + 1)
        taken[y0:y1, x0:x1] = True
        selected.append((x, y, theta))
    return selected

def draw_minutiae_on_image(img, minutiae, map_shape, img_shape, color=(0,0,255)):
    """
    Draws circles and orientation lines for each minutia.
    Args:
        img: np.ndarray, original image (grayscale or RGB)
        minutiae: list of (x, y, theta) in map coordinates
        map_shape: (h_map, w_map)
        img_shape: (h_img, w_img)
    """
    h_map, w_map = map_shape
    h_img, w_img = img_shape
    scale_x = w_img / w_map
    scale_y = h_img / h_map

    # Ensure BGR for cv2
    if len(img.shape) == 2:
        img_vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_vis = img.copy()

    for x, y, theta in minutiae:
        # Map from map coordinates to image coordinates
        ix = int(x * scale_x)
        iy = int(y * scale_y)

        cv2.circle(img_vis, (ix, iy), 4, color, 1)
        # Direction line: add pi/2 so 0 points up
        angle = theta + math.pi/2
        dx = int(15 * math.sin(angle))
        dy = int(15 * math.cos(angle))
        cv2.line(img_vis, (ix, iy), (ix + dx, iy + dy), color, 1)
    return img_vis

def sanity_check_deepprint_sample(dataset, idx=None, threshold=0.5, nms_dist=8):
    """
    Visualize a sample from DeepPrintDataset after all augmentations,
    with extracted minutiae drawn over the cropped image.
    """
    if idx is None:
        idx = np.random.randint(0, len(dataset))
    img, label, H = dataset[idx]  # img: (1,448,448), H: (6,192,192)
    
    # Prepare image for cv2 (uint8, shape HxW)
    img_np = (img.squeeze().numpy() * 255).astype(np.uint8)
    h_img, w_img = img_np.shape
    h_map, w_map = H.shape[1], H.shape[2]
    
    # Extract minutiae from map
    minutiae = extract_minutiae_from_map(H.permute(1,2,0).numpy(), threshold, nms_dist)

    # Draw on image
    img_with_minutiae = draw_minutiae_on_image(img_np, minutiae, map_shape=(h_map, w_map), img_shape=(h_img, w_img))
    
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(img_with_minutiae, cv2.COLOR_BGR2RGB))
    plt.title(f"Augmented sample idx={idx}, label={label}, #minutiae={len(minutiae)}")
    plt.axis('off')
    plt.show()

def save_concat_aligned_by_class(aligned_imgs, labels, outdir, batch_idx):
    """
    Save concatenated images of all aligned samples for each class in the batch.

    Args:
        aligned_imgs: (B, H, W) or (B, 1, H, W) numpy or torch array, aligned images
        labels: (B,) torch tensor or numpy array, class labels
        outdir: str, directory to save images
        batch_idx: int, current batch index (for unique filenames)
    """
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # Ensure numpy (convert from torch if needed)
    if hasattr(aligned_imgs, "detach"):
        aligned_imgs = aligned_imgs.detach().cpu().numpy()
    if hasattr(labels, "detach"):
        labels = labels.detach().cpu().numpy()
    # Remove channel dim if present
    if aligned_imgs.ndim == 4 and aligned_imgs.shape[1] == 1:
        aligned_imgs = aligned_imgs[:, 0, :, :]
    elif aligned_imgs.ndim == 4 and aligned_imgs.shape[-1] == 1:
        aligned_imgs = aligned_imgs[..., 0]
    # Normalize to uint8 [0,255] if not already
    if aligned_imgs.max() <= 1.0:
        aligned_imgs = (aligned_imgs * 255).astype(np.uint8)
    else:
        aligned_imgs = aligned_imgs.astype(np.uint8)

    unique_labels = np.unique(labels)
    for class_idx in unique_labels:
        indices = np.where(labels == class_idx)[0]
        if len(indices) == 0:
            continue
        # Stack aligned images horizontally for this class
        imgs_to_concat = [aligned_imgs[i] for i in indices]
        concat_img = np.concatenate(imgs_to_concat, axis=1)  # H, W1+W2+..., single channel
        # Convert to BGR for saving if grayscale
        if concat_img.ndim == 2:
            concat_img = cv2.cvtColor(concat_img, cv2.COLOR_GRAY2BGR)
        outpath = os.path.join(outdir, f"batch{batch_idx:04d}_class{class_idx}.png")
        cv2.imwrite(outpath, concat_img)