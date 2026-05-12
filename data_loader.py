"""
data_loader.py
--------------
DRIVE dataset loader for retinal vessel segmentation.

CORRECT USAGE:
    images/       -> input retinal images
    1st_manual/   -> vessel segmentation ground truth masks
    mask/         -> field-of-view masks (NOT training labels)

Folder structure:
-----------------
DRIVE/
│
├── training/
│   ├── images/
│   │     ├── 21_training.tif
│   │     └── ...
│   │
│   ├── 1st_manual/
│   │     ├── 21_manual1.gif
│   │     └── ...
│   │
│   └── mask/
│
├── test/
│   ├── images/
│   ├── 1st_manual/
│   └── mask/
"""

from pathlib import Path
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2


# ─────────────────────────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────────────────────────

def get_train_transforms(img_size: int = 256):

    return A.Compose([
        A.Resize(img_size, img_size),

        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),

        A.Rotate(limit=20, p=0.5),

        A.RandomBrightnessContrast(p=0.3),

        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),

        ToTensorV2(),
    ])


def get_val_transforms(img_size: int = 256):

    return A.Compose([
        A.Resize(img_size, img_size),

        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),

        ToTensorV2(),
    ])


# ─────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────

class DRIVEDataset(Dataset):

    def __init__(
        self,
        root: str,
        split: str = "training",
        transform=None,
        img_size: int = 256
    ):

        super().__init__()

        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.img_size = img_size

        # -----------------------------------------------------
        # DIRECTORIES
        # -----------------------------------------------------

        self.image_dir = self.root / split / "images"
        self.mask_dir = self.root / split / "1st_manual"

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")

        # -----------------------------------------------------
        # IMAGE LIST
        # -----------------------------------------------------

        self.image_paths = sorted(list(self.image_dir.glob("*")))

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in {self.image_dir}")

        self.mask_paths = []

        # -----------------------------------------------------
        # MATCH IMAGES ↔ MASKS
        # -----------------------------------------------------

        for img_path in self.image_paths:

            # Example:
            # 21_training.tif -> 21
            idx = img_path.stem.split("_")[0]

            # DRIVE manual mask naming
            mask_name = f"{idx}_manual1.gif"

            mask_path = self.mask_dir / mask_name

            if not mask_path.exists():
                raise FileNotFoundError(
                    f"Mask not found for image:\n"
                    f"Image : {img_path.name}\n"
                    f"Expected mask : {mask_name}"
                )

            self.mask_paths.append(mask_path)

        print(f"[DRIVEDataset] {split}: {len(self.image_paths)} matched pairs loaded.")

    # ---------------------------------------------------------
    # LENGTH
    # ---------------------------------------------------------

    def __len__(self):
        return len(self.image_paths)

    # ---------------------------------------------------------
    # GET ITEM
    # ---------------------------------------------------------

    def __getitem__(self, idx):

        # -----------------------------------------------------
        # LOAD IMAGE
        # -----------------------------------------------------

        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = np.array(image, dtype=np.uint8)

        # -----------------------------------------------------
        # LOAD MASK
        # -----------------------------------------------------

        mask = Image.open(self.mask_paths[idx]).convert("L")
        mask = np.array(mask, dtype=np.uint8)

        # Binary vessel mask
        mask = (mask > 0).astype(np.uint8)

        # -----------------------------------------------------
        # AUGMENTATION
        # -----------------------------------------------------

        if self.transform is not None:

            augmented = self.transform(
                image=image,
                mask=mask
            )

            image = augmented["image"]
            mask = augmented["mask"]

        else:

            image = torch.from_numpy(
                image.transpose(2, 0, 1)
            ).float() / 255.0

            mask = torch.from_numpy(mask)

        # -----------------------------------------------------
        # SHAPES
        # image -> (3,H,W)
        # mask  -> (1,H,W)
        # -----------------------------------------------------

        mask = mask.unsqueeze(0).float()

        return image, mask


# ─────────────────────────────────────────────────────────────
# DATALOADER FACTORY
# ─────────────────────────────────────────────────────────────

def get_dataloaders(
    root: str,
    img_size: int = 256,
    batch_size: int = 4,
    num_workers: int = 0
):

    train_dataset = DRIVEDataset(
        root=root,
        split="training",
        transform=get_train_transforms(img_size),
        img_size=img_size
    )

    val_dataset = DRIVEDataset(
        root=root,
        split="test",
        transform=get_val_transforms(img_size),
        img_size=img_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────
# DEBUG CHECK
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    ROOT = "D:/pillai/DRIVE"

    train_loader, val_loader = get_dataloaders(
        root=ROOT,
        img_size=256,
        batch_size=2
    )

    x, y = next(iter(train_loader))

    print("\nBatch Shapes:")
    print("Images :", x.shape)
    print("Masks  :", y.shape)

    print("\nMask Unique Values:")
    print(torch.unique(y))