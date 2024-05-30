# -*- coding: utf-8 -*-
"""experiments.ipynb

from PIL import Image, ImageEnhance, ImageOps
import matplotlib.pyplot as plt
import mahotas
import numpy as np

def augment_image(image_path):
    # Load the image
    original_image = remove_background(image_path)

    # Display the original image
    plt.subplot(2, 3, 1)
    plt.imshow(original_image, cmap='gray')  # Assuming the original image is grayscale
    plt.title('Original Image')

    # Mirroring (horizontal flip)
    mirrored_image = original_image.transpose(Image.FLIP_LEFT_RIGHT)
    plt.subplot(2, 3, 2)
    plt.imshow(mirrored_image, cmap='gray')
    plt.title('Mirrored Image')

    # Rotation (90 degrees clockwise)
    rotated_image = original_image.rotate(90)
    plt.subplot(2, 3, 3)
    plt.imshow(rotated_image, cmap='gray')
    plt.title('Rotated Image')

    # Flipping (vertical flip)
    flipped_image = original_image.transpose(Image.FLIP_TOP_BOTTOM)
    plt.subplot(2, 3, 4)
    plt.imshow(flipped_image, cmap='gray')
    plt.title('Flipped Image')

    # Color Augmentation (increase brightness)
    enhancer = ImageEnhance.Brightness(original_image)
    brightened_image = enhancer.enhance(1.5)
    plt.subplot(2, 3, 5)
    plt.imshow(brightened_image, cmap='gray')
    plt.title('Brightened Image')

    # Color Augmentation (convert to grayscale)
    grayscale_image = original_image.convert('L')
    plt.subplot(2, 3, 6)
    plt.imshow(grayscale_image, cmap='gray')
    plt.title('Grayscale Image')

    # Display the augmented images
    plt.tight_layout()
    plt.show()

def remove_background(image_path):
    # Load the image
    img = mahotas.imread(image_path)

    # Perform Otsu thresholding to separate foreground and background
    T_otsu = mahotas.otsu(img)
    thresholded_img = img > T_otsu

    # Create a mask to keep the foreground in grayscale and set the background to white
    grayscale_img = img.copy()
    grayscale_img[thresholded_img] = 255  # Set foreground to white

    return Image.fromarray(grayscale_img.astype(np.uint8))

# Example usage:
image_path = '/content/original_46_1.png'
augment_image(image_path)

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt

def resize_images(img1, img2):
    # Resize images to the minimum dimensions
    min_height = min(img1.shape[0], img2.shape[0])
    min_width = min(img1.shape[1], img2.shape[1])

    img1_resized = cv2.resize(img1, (min_width, min_height))
    img2_resized = cv2.resize(img2, (min_width, min_height))

    return img1_resized, img2_resized

def calculate_scores(image_path1, image_path2):
    # Read images
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)

    # Resize images to have the same dimensions
    img1, img2 = resize_images(img1, img2)

    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Calculate SSIM
    ssim_score, _ = ssim(gray1, gray2, full=True)

    # Calculate PSNR
    psnr_score = psnr(gray1, gray2)

    # Calculate pixel-wise difference (Pix2Pix difference)
    pixel_diff = np.sum(np.abs(img1 - img2))

    return ssim_score, psnr_score, pixel_diff, img1, img2

# Example usage:
image_path1 = '/content/original_3_7.png'
image_path2 = '/content/original_3_7.png.png'

ssim_score, psnr_score, pixel_diff, img1, img2 = calculate_scores(image_path1, image_path2)

# Plot the images with scores
plt.figure(figsize=(12, 6))

# Original Image 1
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
plt.title('Generated Image')

# Original Image 2
plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

# Display Scores
plt.subplot(1, 3, 3)
plt.text(0.5, 0.5, f"SSIM: {ssim_score:.4f}\nPSNR: {psnr_score:.4f}\nPixel Diff: {pixel_diff}",
         fontsize=12, ha='center', va='center')
plt.axis('off')
plt.title('Scores')

plt.show()
print(f"SSIM: {ssim_score:.4f}\nPSNR: {psnr_score:.4f}\nPixel Diff: {pixel_diff}")

!pip install mahotas

import os
import mahotas
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from PIL import Image

def remove_background(image_path):
    # Load the image
    img = mahotas.imread(image_path)

    # Perform Otsu thresholding to separate foreground and background
    T_otsu = mahotas.otsu(img)
    thresholded_img = img > T_otsu

    # Create a mask to keep the foreground in grayscale and set the background to white
    grayscale_img = img.copy()
    grayscale_img[thresholded_img] = 255  # Set foreground to white

    # Plot the original and processed images side by side
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Plot original image
    axs[0].imshow(img, cmap='gray')
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    # Plot image after background removal
    axs[1].imshow(grayscale_img, cmap='gray')
    axs[1].set_title('After Background Removal')
    axs[1].axis('off')

    plt.show()

    return grayscale_img.astype(np.uint8)

# Example usage
image_path = "/content/original_1_1.png"
result_img = remove_background(image_path)

