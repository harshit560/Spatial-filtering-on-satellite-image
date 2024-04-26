#!/usr/bin/env python


import cv2
import numpy as np
import matplotlib.pyplot as plt
import rasterio

# Load the satellite image
image_path = "C:\\Users\\Hp\\Desktop\\soul\\image.tif"
with rasterio.open(image_path) as src:
    gray_image = src.read(1)
    transform = src.transform
    extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

if gray_image is None:
    print("Error: Unable to load the image.")
else:
    # Convert the image data type to uint8
    gray_image_uint8 = (gray_image / np.max(gray_image) * 255).astype(np.uint8)

    # Display the original grayscale image
    plt.figure(figsize=(8, 8))
    plt.subplot(3, 3, 1)
    plt.imshow(gray_image_uint8, cmap='gray', extent=extent)
    plt.title('Original Grayscale Image')

    # Define kernel sizes for filtering according to your use
    kernel_size = 3

    # Smoothing Filters
    smoothed_image = cv2.blur(gray_image_uint8, (kernel_size, kernel_size))
    with rasterio.open("C:\\Users\\Hp\\Desktop\\soul\\smoothed_image.tif", 'w', driver='GTiff', width=smoothed_image.shape[1], height=smoothed_image.shape[0], count=1, dtype=str(smoothed_image.dtype), crs=src.crs, transform=transform) as dst:
        dst.write(smoothed_image, 1)
    plt.subplot(3, 3, 2)
    plt.imshow(smoothed_image, cmap='gray', extent=extent)
    plt.title('Mean Filter')

    # Sharpening Filters
    sharpened_image = cv2.filter2D(gray_image_uint8, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))
    with rasterio.open("C:\\Users\\Hp\\Desktop\\soul\\sharpened_image.tif", 'w', driver='GTiff', width=sharpened_image.shape[1], height=sharpened_image.shape[0], count=1, dtype=str(sharpened_image.dtype), crs=src.crs, transform=transform) as dst:
        dst.write(sharpened_image, 1)
    plt.subplot(3, 3, 3)
    plt.imshow(sharpened_image, cmap='gray', extent=extent)
    plt.title('Laplacian Filter')

    # Edge Detection Filters
    sobel_x = cv2.Sobel(gray_image_uint8, cv2.CV_64F, 1, 0, ksize=kernel_size).astype(np.uint8)  # Convert to uint8
    sobel_y = cv2.Sobel(gray_image_uint8, cv2.CV_64F, 0, 1, ksize=kernel_size).astype(np.uint8)  # Convert to uint8
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2).astype(np.uint8)  # Convert to uint8
    with rasterio.open("C:\\Users\\Hp\\Desktop\\soul\\sobel_mag.tif", 'w', driver='GTiff', width=sobel_mag.shape[1], height=sobel_mag.shape[0], count=1, dtype=str(sobel_mag.dtype), crs=src.crs, transform=transform) as dst:
        dst.write(sobel_mag, 1)
    plt.subplot(3, 3, 4)
    plt.imshow(sobel_mag, cmap='gray', extent=extent)
    plt.title('Sobel Filter')

    # Gradient Filters
    gradient_mag = cv2.Laplacian(gray_image_uint8, cv2.CV_64F, ksize=kernel_size).astype(np.uint8)  # Convert to uint8
    with rasterio.open("C:\\Users\\Hp\\Desktop\\soul\\gradient_mag.tif", 'w', driver='GTiff', width=gradient_mag.shape[1], height=gradient_mag.shape[0], count=1, dtype=str(gradient_mag.dtype), crs=src.crs, transform=transform) as dst:
        dst.write(gradient_mag, 1)
    plt.subplot(3, 3, 5)
    plt.imshow(np.abs(gradient_mag), cmap='gray', extent=extent)
    plt.title('Gradient Magnitude')

    # Morphological Filters
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    erosion = cv2.erode(gray_image_uint8, kernel, iterations=1)
    with rasterio.open("C:\\Users\\Hp\\Desktop\\soul\\erosion.tif", 'w', driver='GTiff', width=erosion.shape[1], height=erosion.shape[0], count=1, dtype=str(erosion.dtype), crs=src.crs, transform=transform) as dst:
        dst.write(erosion, 1)
    plt.subplot(3, 3, 6)
    plt.imshow(erosion, cmap='gray', extent=extent)
    plt.title('Erosion')

    dilation = cv2.dilate(gray_image_uint8, kernel, iterations=1)
    with rasterio.open("C:\\Users\\Hp\\Desktop\\soul\\dilation.tif", 'w', driver='GTiff', width=dilation.shape[1], height=dilation.shape[0], count=1, dtype=str(dilation.dtype), crs=src.crs, transform=transform) as dst:
        dst.write(dilation, 1)
    plt.subplot(3, 3, 7)
    plt.imshow(dilation, cmap='gray', extent=extent)
    plt.title('Dilation')

    opening = cv2.morphologyEx(gray_image_uint8, cv2.MORPH_OPEN, kernel)
    with rasterio.open("C:\\Users\\Hp\\Desktop\\soul\\opening.tif", 'w', driver='GTiff', width=opening.shape[1], height=opening.shape[0], count=1, dtype=str(opening.dtype), crs=src.crs, transform=transform) as dst:
        dst.write(opening, 1)
    plt.subplot(3, 3, 8)
    plt.imshow(opening, cmap='gray', extent=extent)
    plt.title('Opening')

    closing = cv2.morphologyEx(gray_image_uint8, cv2.MORPH_CLOSE, kernel)
    with rasterio.open("C:\\Users\\Hp\\Desktop\\soul\\closing.tif", 'w', driver='GTiff', width=closing.shape[1], height=closing.shape[0], count=1, dtype=str(closing.dtype), crs=src.crs, transform=transform) as dst:
        dst.write(closing, 1)
    plt.subplot(3, 3, 9)
    plt.imshow(closing, cmap='gray', extent=extent)
    plt.title('Closing')

    plt.tight_layout()
    plt.show()

