import torch
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os
import matplotlib.pyplot as plt

class FundusDataset(Dataset):
    def __init__(self, img_dir, mask_dir, augment=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.files = sorted(os.listdir(img_dir))
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = np.array(Image.open(img_path).convert("RGB"))
        img = img[:, :, 1]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img = clahe.apply(img)
        img = img.astype(np.float32) / 255.0

        base_name, ext = os.path.splitext(img_name)
        mask_candidates = [f for f in os.listdir(self.mask_dir) if f.startswith(base_name)]
        if len(mask_candidates) == 0:
            raise FileNotFoundError(f"No mask found for {img_name}")
        mask_path = os.path.join(self.mask_dir, mask_candidates[0])
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = (mask > 127).astype(np.float32)

        if self.augment:
            augmented = self.augment(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        else:
            img = torch.tensor(img).unsqueeze(0)
            mask = torch.tensor(mask).unsqueeze(0)

        return img, mask

augmentations = A.Compose([
    A.Resize(512,512),
    A.Rotate(limit=10, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15, rotate_limit=0, p=0.5),
    A.RandomCrop(width=480, height=480, p=0.5),
    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=10, p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
    A.GammaCorrection(gamma_limit=(80,120), p=0.5),
    A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0, hue=0, p=0.5),
    A.GaussNoise(var_limit=(5.0, 20.0), p=0.3),
    A.Sharpen(alpha=(0.2,0.5), lightness=(0.5,1.0), p=0.3),
    ToTensorV2()
])

train_ds = FundusDataset("dataset/images", "dataset/masks", augment=augmentations)
train_dl = DataLoader(train_ds, batch_size=2, shuffle=True)

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=1,
    classes=1
)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

loss_fn = smp.losses.DiceLoss(mode='binary')
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for epoch in range(20):
    model.train()
    running_loss = 0
    for imgs, masks in train_dl:
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs)
        loss = loss_fn(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {running_loss/len(train_dl):.4f}")

model.eval()
with torch.no_grad():
    imgs, masks = next(iter(train_dl))
    imgs = imgs.to(device)
    preds = model(imgs)
    preds = (torch.sigmoid(preds) > 0.5).cpu().numpy()

    for i in range(len(imgs)):
        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1)
        plt.title("Input (green channel)")
        plt.imshow(imgs[i].cpu().squeeze(), cmap='gray')
        plt.subplot(1,3,2)
        plt.title("Ground Truth")
        plt.imshow(masks[i].squeeze(), cmap='gray')
        plt.subplot(1,3,3)
        plt.title("Prediction")
        plt.imshow(preds[i][0], cmap='gray')
        plt.show()
