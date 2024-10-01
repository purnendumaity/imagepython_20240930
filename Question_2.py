# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 10:31:29 2024
@author: purnendumaity@gmail.com
Implement both the Floyd-Steinberg and Jarvis-Judice-Ninke dithering algorithms on
the image in Python,then compare the results obtained from each method
package needed: numpy, pillow, matplotlib
"""

import numpy as np
from PIL import Image
import matplotlib
#working in Pycharm community version
#so below 2 line adjusted to see my plot window one by one
matplotlib.use('TkAgg',force=True)
print("Switched to:",matplotlib.get_backend())
import matplotlib.pyplot as plt

# Define the Dithering Algorithms
# Floyd - SteinbergDithering:
def floyd_steinberg_dithering(img):
    # Define the error diffusion matrix for Floyd-Steinberg Dithering
    fs_matrix = np.array([[0, 0, 7],
                          [3, 5, 1]]) / 16.0

    img = img.copy()
    rows, cols = img.shape

    for row in range(rows):
        for col in range(cols):
            old_pixel = img[row, col]
            new_pixel = np.round(old_pixel / 255.0) * 255
            img[row, col] = new_pixel
            quant_error = old_pixel - new_pixel

            # Distribute the error to neighboring pixels using the FS matrix
            for i in range(2):
                for j in range(3):
                    if row + i < rows and col + j - 1 < cols and col + j - 1 >= 0:
                        img[row + i, col + j - 1] += quant_error * fs_matrix[i, j]

    return img


#Jarvis - Judice - Ninke Dithering:
def jarvis_judice_ninke_dithering(img):
    # Define the error diffusion matrix for Jarvis-Judice-Ninke Dithering
    jjn_matrix = np.array([[0, 0, 0, 7, 5],
                           [3, 5, 7, 5, 3],
                           [1, 3, 5, 3, 1]]) / 48.0

    img = img.copy()
    rows, cols = img.shape

    for row in range(rows):
        for col in range(cols):
            old_pixel = img[row, col]
            new_pixel = np.round(old_pixel / 255.0) * 255
            img[row, col] = new_pixel
            quant_error = old_pixel - new_pixel

            # Distribute the error to neighboring pixels using the JJN matrix
            for i in range(3):
                for j in range(5):
                    if row + i < rows and col + j - 2 < cols and col + j - 2 >= 0:
                        img[row + i, col + j - 2] += quant_error * jjn_matrix[i, j]

    return img

#Load and Prepare the Image:
def load_image_grayscale(image_path):
    # Open image and convert to grayscale
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = np.array(img, dtype=np.float64)  # Convert to float64 for precision
    return img

#Dispaly Results
def display_results(original_img, floyd_img, jjn_img):
    # Display the original and dithered images
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 3, 1)
    plt.imshow(original_img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(floyd_img, cmap='gray')
    plt.title('Floyd-Steinberg Dithering')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(jjn_img, cmap='gray')
    plt.title('Jarvis-Judice-Ninke Dithering')
    plt.axis('off')
    plt.show()

#MainScript:
# Load and process the image
image_path = "./humanface.jpg" # Replace with the actual path to your image
img = load_image_grayscale(image_path)
# Apply Floyd-Steinberg Dithering
floyd_img = floyd_steinberg_dithering(img)
# Apply Jarvis-Judice-Ninke Dithering
jjn_img = jarvis_judice_ninke_dithering(img)
# Compare the results
display_results(img, floyd_img, jjn_img)
