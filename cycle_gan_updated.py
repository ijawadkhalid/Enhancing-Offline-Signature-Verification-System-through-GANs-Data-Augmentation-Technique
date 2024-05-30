# -*- coding: utf-8 -*-
"""Cycle GAN updated.ipynb

1. Install all dependencies
"""

!pip install mahotas

"""2. Upload dataset"""

import os
import gdown
from pathlib import Path

"""3. Resize all images"""

import os
from PIL import Image

def resize_and_replace_images(directory_path, target_size=(224, 224)):
    """
    Resize all images in the specified directory to the target size and convert them to 1 channel (grayscale).
    The original images will be replaced.

    Parameters:
    - directory_path (str): Path to the directory containing the images.
    - target_size (tuple): The target size (width, height) to which the images will be resized.

    Returns:
    None
    """
    # List all files in the directory
    files = os.listdir(directory_path)

    for file_name in files:
        file_path = os.path.join(directory_path, file_name)

        # Check if the file is an image (you can add more image extensions if needed)
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
            try:
                # Open the image
                img = Image.open(file_path)

                # Resize the image to the target size
                img = img.resize(target_size, Image.ANTIALIAS)

                # Convert the image to grayscale (1 channel)
                img = img.convert('L')

                # Replace the original image with the resized and converted one
                img.save(file_path)
                print(f"Resized and replaced {file_name}")

            except Exception as e:
                print(f"Error processing {file_name}: {e}")

resize_and_replace_images('data/Individual signature/testA', target_size=(224, 224))
resize_and_replace_images('data/Individual signature/testB', target_size=(224, 224))
resize_and_replace_images('data/Individual signature/trainA', target_size=(224, 224))
resize_and_replace_images('data/Individual signature/trainB', target_size=(224, 224))

import os
from PIL import Image
import numpy as np

def print_image_shapes(directory_path):
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)

        if os.path.isfile(file_path) and file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            try:
                image = Image.open(file_path)
                image_array = np.array(image)
                print(f"Image: {filename}, Shape: {image_array.shape}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

# Example usage
directory_path = "data/Individual signature/testA"
print_image_shapes(directory_path)

"""4. All configuration"""

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "data/Individual signature/train"
VAL_DIR = "data/Individual signature/val"
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 1
NUM_EPOCHS = 100
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN_F = "genf.pth.tar"
CHECKPOINT_GEN_O = "geno.pth.tar"
CHECKPOINT_CRITIC_F = "criticf.pth.tar"
CHECKPOINT_CRITIC_O = "critico.pth.tar"

transforms = A.Compose(
    [
        # A.Resize(width=224, height=224),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5], std=[0.5], max_pixel_value=255),  # Assuming single-channel grayscale
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)

import random, torch, os, numpy as np
import torch.nn as nn
import copy

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

"""Dataset"""

from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np
import mahotas
import albumentations as A
from albumentations.pytorch import ToTensorV2

def remove_background(image_path):
    # Load the image
    img = mahotas.imread(image_path)

    # Perform Otsu thresholding to separate foreground and background
    T_otsu = mahotas.otsu(img)
    thresholded_img = img > T_otsu

    # Create a mask to keep the foreground in grayscale and set the background to white
    grayscale_img = img.copy()
    grayscale_img[thresholded_img] = 255  # Set foreground to white

    return grayscale_img.astype(np.uint8)

class SignatureDataset(Dataset):
    def __init__(self, root_org, root_forg, transform=None):
        self.root_org = root_org
        self.root_forg = root_forg
        self.transform = transform

        self.org_img = os.listdir(root_org)
        self.forg_img = os.listdir(root_forg)
        self.length_dataset = max(len(self.org_img), len(self.forg_img))
        self.org_len = len(self.org_img)
        self.forg_len = len(self.forg_img)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        org_img = self.org_img[index % self.org_len]
        forg_img = self.forg_img[index % self.forg_len]

        org_path = os.path.join(self.root_org, org_img)
        forg_path = os.path.join(self.root_forg, forg_img)

        org_img = remove_background(org_path)
        forg_img = remove_background(forg_path)

        if self.transform:
            augmentations = self.transform(image=org_img, image0=forg_img)
            org_img = augmentations["image"]
            forg_img = augmentations["image0"]

        return org_img, forg_img, org_path, forg_path

"""5. Discriminator Architecture

"""

import torch
import torch.nn as nn
from torchsummary import summary


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                4,
                stride,
                1,
                bias=True,
                padding_mode="reflect",
            ),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                Block(in_channels, feature, stride=1 if feature == features[-1] else 2)
            )
            in_channels = feature
        layers.append(
            nn.Conv2d(
                in_channels,
                1,
                kernel_size=4,
                stride=1,
                padding=1,
                padding_mode="reflect",
            )
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial(x)
        return torch.sigmoid(self.model(x))


def test():
    x = torch.randn((5, 1, 256, 256))
    model = Discriminator(in_channels=1)
    preds = model(x)
    print(preds.shape)


if __name__ == "__main__":
    test()

"""6. Generator Architecture"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs)
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity(),
        )

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1),
            ConvBlock(channels, channels, use_act=False, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, img_channels, num_features=64, num_residuals=9):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                img_channels,
                num_features,
                kernel_size=7,
                stride=1,
                padding=3,
                padding_mode="reflect",
            ),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True),
        )
        self.down_blocks = nn.ModuleList(
            [
                ConvBlock(
                    num_features, num_features * 2, kernel_size=3, stride=2, padding=1
                ),
                ConvBlock(
                    num_features * 2,
                    num_features * 4,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            ]
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_features * 4) for _ in range(num_residuals)]
        )
        self.up_blocks = nn.ModuleList(
            [
                ConvBlock(
                    num_features * 4,
                    num_features * 2,
                    down=False,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                ConvBlock(
                    num_features * 2,
                    num_features * 1,
                    down=False,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
            ]
        )

        self.last = nn.Conv2d(
            num_features * 1,
            img_channels,
            kernel_size=7,
            stride=1,
            padding=3,
            padding_mode="reflect",
        )

    def forward(self, x):
        x = self.initial(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.res_blocks(x)
        for layer in self.up_blocks:
            x = layer(x)
        return torch.tanh(self.last(x))

def test():
  img_channels = 1
  img_size = 256
  x = torch.randn((2, img_channels, img_size, img_size))
  gen = Generator(img_channels, 9)
  print(gen(x).shape)


if __name__ == "__main__":
    test()

"""Train model"""

import torch
import sys
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision.utils import save_image


def train_fn(
    disc_forg, disc_org, gen_org, gen_forg, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler
):
    F_real = 0
    F_fake = 0
    loop = tqdm(loader, leave=True)

    for idx, (org, forg, org_path, forg_path) in enumerate(loop):
        org = org.to(DEVICE)
        forg = forg.to(DEVICE)

        # print(f"Original Image {org.shape}")
        # print(f"Forgery Image {forg.shape}")
        # print(f"Original Path {org_path}")
        # print(f"Forged Path {forg_path}")
        # Train Discriminators H and Z
        with torch.cuda.amp.autocast():
            fake_forg = gen_forg(org)
            D_F_real = disc_forg(forg)
            D_F_fake = disc_forg(fake_forg.detach())
            F_real += D_F_real.mean().item()
            F_fake += D_F_fake.mean().item()
            D_F_real_loss = mse(D_F_real, torch.ones_like(D_F_real))
            D_F_fake_loss = mse(D_F_fake, torch.zeros_like(D_F_fake))
            D_F_loss = D_F_real_loss + D_F_fake_loss

            fake_org = gen_org(forg)
            D_O_real = disc_org(org)
            D_O_fake = disc_org(fake_org.detach())
            D_O_real_loss = mse(D_O_real, torch.ones_like(D_O_real))
            D_O_fake_loss = mse(D_O_fake, torch.zeros_like(D_O_fake))
            D_O_loss = D_O_real_loss + D_O_fake_loss

            # put it togethor
            D_loss = (D_F_loss + D_O_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators H and Z
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_F_fake = disc_forg(fake_forg)
            D_O_fake = disc_org(fake_org)
            loss_G_H = mse(D_F_fake, torch.ones_like(D_F_fake))
            loss_G_Z = mse(D_O_fake, torch.ones_like(D_O_fake))

            # cycle loss
            cycle_org = gen_org(fake_forg)
            cycle_forg = gen_forg(fake_org)
            cycle_org_loss = l1(org, cycle_org)
            cycle_forg_loss = l1(forg, cycle_forg)

            # identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_org = gen_org(org)
            identity_forg = gen_forg(forg)
            identity_org_loss = l1(org, identity_org)
            identity_forg_loss = l1(forg, identity_forg)

            # add all togethor
            G_loss = (
                loss_G_Z
                + loss_G_H
                + cycle_org_loss * LAMBDA_CYCLE
                + cycle_forg_loss * LAMBDA_CYCLE
                + identity_forg_loss * LAMBDA_IDENTITY
                + identity_org_loss * LAMBDA_IDENTITY
            )
        print("Loss: " + str(G_loss))
        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 200 == 0:
            # save_image(fake_forg * 0.5 + 0.5, f"saved_images/org_{idx}.png")
            # save_image(fake_org * 0.5 + 0.5, f"saved_images/forg{idx}.png")
            save_image(fake_forg * 0.5 + 0.5, f"saved_images/{os.path.basename(str(org_path[0]))}.png")
            save_image(fake_org * 0.5 + 0.5, f"saved_images/{os.path.basename(str(forg_path[0]))}.png")

        loop.set_postfix(H_real=F_real / (idx + 1), H_fake=F_fake / (idx + 1))


def main():
    disc_forg = Discriminator(in_channels=1).to(DEVICE)
    disc_org = Discriminator(in_channels=1).to(DEVICE)
    gen_org = Generator(img_channels=1, num_residuals=9).to(DEVICE)
    gen_forg = Generator(img_channels=1, num_residuals=9).to(DEVICE)

    opt_disc = optim.Adam(
        list(disc_forg.parameters()) + list(disc_org.parameters()),
        lr=LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_org.parameters()) + list(gen_forg.parameters()),
        lr=LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if LOAD_MODEL:
        load_checkpoint(
            CHECKPOINT_GEN_F,
            gen_forg,
            opt_gen,
            LEARNING_RATE,
        )
        load_checkpoint(
            CHECKPOINT_GEN_O,
            gen_org,
            opt_gen,
            LEARNING_RATE,
        )
        load_checkpoint(
            CHECKPOINT_CRITIC_F,
            disc_forg,
            opt_disc,
            LEARNING_RATE,
        )
        load_checkpoint(
            CHECKPOINT_CRITIC_O,
            disc_org,
            opt_disc,
            LEARNING_RATE,
        )

    dataset = SignatureDataset(
        root_forg="data/Individual signature/trainA",
        root_org="data/Individual signature/trainB",
        transform=transforms,
    )
    val_dataset = SignatureDataset(
        root_forg="data/Individual signature/testA",
        root_org="data/Individual signature/testB",
        transform=transforms,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        train_fn(
            disc_forg,
            disc_org,
            gen_org,
            gen_forg,
            loader,
            opt_disc,
            opt_gen,
            L1,
            mse,
            d_scaler,
            g_scaler,
        )

        if SAVE_MODEL:
            save_checkpoint(gen_forg, opt_gen, filename=CHECKPOINT_GEN_F)
            save_checkpoint(gen_org, opt_gen, filename=CHECKPOINT_GEN_O)
            save_checkpoint(disc_forg, opt_disc, filename=CHECKPOINT_GEN_F)
            save_checkpoint(disc_org, opt_disc, filename=CHECKPOINT_GEN_O)

if __name__ == "__main__":
    main()

import matplotlib.pyplot as plt

hist = [19.9301, 17.5212, 14.2457, 11.5093, 9.3707, 7.7806, 6.6840, 5.7951, 5.1128, 4.6647, 4.2299, 3.8405, 3.5718, 3.3266, 3.1247, 2.9386, 2.6429, 2.6160, 2.3284, 2.1780, 2.3118, 2.0963, 1.9659, 2.0250, 1.9554, 1.8291, 1.7746, 1.7371, 1.8633, 1.5998, 1.7720, 1.6790, 1.5300, 1.5577, 1.4419, 1.4457, 1.4720, 1.5102, 1.2880, 1.3655, 1.3713, 1.2315, 1.0602, 1.0988, 1.1965, 0.9893]

# Create x-axis values (epochs)
epochs = range(1, len(hist) + 1)

# Plot the CycleGAN loss graph
plt.plot(epochs, hist, marker='o', linestyle='-', color='b')
plt.title('CycleGAN Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

