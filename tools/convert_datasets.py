import os
import glob
import shutil
from collections import defaultdict
from pathlib import Path
from PIL import Image

def prepare_fingerprint_dataset(sources, output_dir='dataset', min_samples=4):
    img_out_dir = Path(output_dir) / 'images'
    img_out_dir.mkdir(parents=True, exist_ok=True)

    finger_id = 0

    for name, path in sources.items():
        print(f"Processing {name}...")

        for dbx in sorted(Path(path).rglob('*')):
            if not dbx.is_dir():
               continue

            groups = defaultdict(list)
            for img_path in sorted(dbx.glob('*.tif')) + sorted(dbx.glob('*.bmp')):
                base = img_path.stem
                if name.lower() == 'neurocrossmatch':
                    parts = base.split('_')
                    if len(parts) >= 2:
                        fid = f"{parts[0]}_{parts[1]}"  # e.g., 012_3
                    else:
                        fid = base
                else:
                    parts = base.split('_')
                    fid = parts[0] if len(parts) > 1 else base
                groups[fid].append(img_path)

            for fid, files in groups.items():
                if len(files) < min_samples:
                    continue
                for i, f in enumerate(sorted(files)):
                    new_name = f"f{finger_id:05d}_{i}.png"
                    new_path = img_out_dir / new_name

                    img = Image.open(f).convert('L')
                    img.save(new_path)

                finger_id += 1

    print(f"✅ Done: {finger_id} identities written to {img_out_dir}")


# ===================== Example Usage =====================
if __name__ == '__main__':
    sources = {
        'FVC2000': '/home/nbeleski/development/datasets/FVC2000',
        'NeuroCrossmatch': '/home/nbeleski/development/datasets/NeuroCrossmatch'
    }
    prepare_fingerprint_dataset(sources, output_dir='dataset', min_samples=4)

