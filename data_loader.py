"""
data_loader.py
Handles everything related to loading and preparing the dataset.

We support DUTS, ECSSD, MSRA10K, and any generic image/mask folder layout.
Augmentations are only applied during training — validation and test sets
always get the clean, unmodified images so metrics are comparable.
"""

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class SODDataset(Dataset):
    """
    Pairs up images with their saliency masks and serves them as tensors.

    During training, we apply a bunch of random augmentations to artificially
    grow the effective dataset size — especially important since ECSSD only
    has 1000 images total. The same random transform is always applied to both
    the image AND the mask together, so they stay aligned.
    """

    def __init__(self, image_paths, mask_paths, img_size=128, augment=False):
        assert len(image_paths) == len(mask_paths)
        self.image_paths = image_paths
        self.mask_paths  = mask_paths
        self.img_size    = img_size
        self.augment     = augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # OpenCV loads BGR by default, so we convert to RGB immediately
        image = cv2.imread(str(self.image_paths[idx]))
        if image is None:
            raise FileNotFoundError(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Masks are single-channel grayscale
        mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(self.mask_paths[idx])

        # Resize everything to the same square size the model expects
        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask  = cv2.resize(mask,  (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        # Normalise image to [0, 1] and binarise the mask
        image = image.astype(np.float32) / 255.0
        mask  = (mask.astype(np.float32) / 255.0 > 0.5).astype(np.float32)

        if self.augment:
            image, mask = self._augment(image, mask)

        # PyTorch wants channels first: (H, W, C) -> (C, H, W)
        image = torch.from_numpy(image.transpose(2, 0, 1))
        mask  = torch.from_numpy(mask[np.newaxis])
        return image, mask

    def _augment(self, image, mask):
        """Apply random transforms. Each one is independent with its own probability."""

        if random.random() > 0.5:
            image = np.fliplr(image).copy()
            mask  = np.fliplr(mask).copy()

        if random.random() > 0.5:
            image = np.flipud(image).copy()
            mask  = np.flipud(mask).copy()

        # Rotation helps the model generalise to objects at different angles
        if random.random() > 0.5:
            angle = random.uniform(-30, 30)
            image, mask = self._rotate(image, mask, angle)

        # Zooming in on a random region simulates different object scales
        if random.random() > 0.5:
            image, mask = self._random_crop(image, mask, min_ratio=0.70)

        if random.random() > 0.5:
            image = np.clip(image * random.uniform(0.6, 1.4), 0, 1)

        if random.random() > 0.5:
            mean  = image.mean()
            image = np.clip((image - mean) * random.uniform(0.6, 1.4) + mean, 0, 1)

        # Saturation jitter is done in HSV space so hue stays consistent
        if random.random() > 0.5:
            image = self._saturation_jitter(image, random.uniform(0.5, 1.5))

        if random.random() > 0.6:
            image = np.clip(image + np.random.normal(0, 0.025, image.shape).astype(np.float32), 0, 1)

        # Random erasing: occlude small patches to simulate partially hidden objects.
        # Only applied to the image, not the mask, since the salient region is still there.
        if random.random() > 0.7:
            image = self._random_erase(image)

        return image, mask

    @staticmethod
    def _rotate(image, mask, angle):
        h, w  = image.shape[:2]
        M     = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        # BORDER_REFLECT_101 fills the corners by mirroring pixels — avoids black borders
        image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT_101)
        mask  = cv2.warpAffine(mask,  M, (w, h), flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_REFLECT_101)
        mask  = (mask > 0.5).astype(np.float32)
        return image, mask

    def _random_crop(self, image, mask, min_ratio=0.70):
        h, w   = image.shape[:2]
        ratio  = random.uniform(min_ratio, 1.0)
        ch, cw = int(h * ratio), int(w * ratio)
        top    = random.randint(0, h - ch)
        left   = random.randint(0, w - cw)
        image  = image[top:top + ch, left:left + cw]
        mask   = mask[top:top + ch, left:left + cw]
        image  = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask   = cv2.resize(mask,  (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        mask   = (mask > 0.5).astype(np.float32)
        return image, mask

    @staticmethod
    def _saturation_jitter(image, factor):
        # Convert to HSV, scale the S channel, convert back
        hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * factor, 0, 255)
        rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return rgb.astype(np.float32) / 255.0

    @staticmethod
    def _random_erase(image, sl=0.02, sh=0.20):
        h, w  = image.shape[:2]
        area  = h * w
        img   = image.copy()
        # Erase 1 to 3 random rectangular patches
        for _ in range(random.randint(1, 3)):
            erase_area = random.uniform(sl, sh) * area
            ratio      = random.uniform(0.3, 3.0)
            eh = min(int((erase_area * ratio) ** 0.5), h)
            ew = min(int((erase_area / ratio) ** 0.5), w)
            top  = random.randint(0, h - eh)
            left = random.randint(0, w - ew)
            img[top:top + eh, left:left + ew] = random.random()
        return img


# ── Finding image/mask paths ──────────────────────────────────────────────────

def load_dataset_paths(data_dir, dataset_type="generic"):
    """
    Figures out where the images and masks live for the given dataset format,
    then returns two matched lists of paths (one per dataset file).
    """
    data_dir = Path(data_dir)

    if dataset_type == "duts":
        # DUTS can be stored a few different ways depending on how it was extracted
        candidates = [
            (data_dir / "DUTS-TR" / "DUTS-TR-Image", data_dir / "DUTS-TR" / "DUTS-TR-Mask"),
            (data_dir / "DUTS-TR-Image",              data_dir / "DUTS-TR-Mask"),
            (data_dir / "DUTS-TE" / "DUTS-TE-Image", data_dir / "DUTS-TE" / "DUTS-TE-Mask"),
        ]
        img_dir, mask_dir = _first_existing_pair(candidates)

    elif dataset_type == "ecssd":
        img_dir, mask_dir = data_dir / "images", data_dir / "ground_truth_mask"

    elif dataset_type == "msra10k":
        img_dir, mask_dir = data_dir / "images", data_dir / "masks"

    else:
        # Generic mode: try common folder names and pick the first match
        for name in ("images", "Images", "img", "Imgs", "JPEGImages"):
            if (data_dir / name).exists():
                img_dir = data_dir / name
                break
        else:
            img_dir = data_dir

        for name in ("masks", "Masks", "mask", "ground_truth_mask", "annotations"):
            if (data_dir / name).exists():
                mask_dir = data_dir / name
                break
        else:
            mask_dir = data_dir

    return _match_pairs(img_dir, mask_dir)


def _first_existing_pair(candidates):
    for pair in candidates:
        if pair[0].exists() and pair[1].exists():
            return pair
    return candidates[0]


def _match_pairs(img_dir, mask_dir):
    """Match images to masks by filename stem (ignoring extension)."""
    IMG_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".tif"}
    MASK_EXTS = [".png", ".jpg", ".bmp"]
    imgs, msks = [], []
    for img_path in sorted(f for f in img_dir.iterdir() if f.suffix.lower() in IMG_EXTS):
        for ext in MASK_EXTS:
            mp = mask_dir / (img_path.stem + ext)
            if mp.exists():
                imgs.append(img_path)
                msks.append(mp)
                break
    return imgs, msks


# ── Building the DataLoaders ──────────────────────────────────────────────────

def get_dataloaders(
    data_dir,
    dataset_type="generic",
    img_size=128,
    batch_size=16,
    num_workers=0,
    train_ratio=0.70,
    val_ratio=0.15,
    random_seed=42,
):
    """
    Splits the dataset and returns three DataLoaders: train, val, test.

    We fix the random seed so the splits are always the same — important
    for fair comparison between runs. Augmentation is enabled only for
    the training loader.
    """
    image_paths, mask_paths = load_dataset_paths(data_dir, dataset_type)
    if not image_paths:
        raise ValueError(f"No paired image-mask files found in '{data_dir}'.")

    print(f"[DataLoader] Found {len(image_paths)} image-mask pairs.")

    indices    = list(range(len(image_paths)))
    test_ratio = 1.0 - train_ratio - val_ratio

    train_idx, temp_idx = train_test_split(indices, test_size=1 - train_ratio, random_state=random_seed)
    val_idx, test_idx   = train_test_split(temp_idx, test_size=test_ratio / (val_ratio + test_ratio), random_state=random_seed)

    print(f"[DataLoader] Split -> Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    def _make_dataset(idx, augment):
        return SODDataset([image_paths[i] for i in idx],
                          [mask_paths[i]  for i in idx],
                          img_size=img_size, augment=augment)

    shared = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return (
        DataLoader(_make_dataset(train_idx, augment=True),  shuffle=True,  **shared),
        DataLoader(_make_dataset(val_idx,   augment=False), shuffle=False, **shared),
        DataLoader(_make_dataset(test_idx,  augment=False), shuffle=False, **shared),
    )


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--dataset_type", default="generic")
    p.add_argument("--img_size", type=int, default=128)
    args = p.parse_args()

    tl, vl, tel = get_dataloaders(args.data_dir, args.dataset_type, args.img_size, batch_size=4)
    imgs, masks = next(iter(tl))
    print(f"Image batch: {imgs.shape}  Mask batch: {masks.shape}")
    print(f"Image range [{imgs.min():.2f}, {imgs.max():.2f}]  Mask range [{masks.min():.2f}, {masks.max():.2f}]")
