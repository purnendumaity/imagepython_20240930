# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 10:31:29 2024
@author: purnendumaity@gmail.com
Take any image and apply the Fourier Transform to this image and the following
filters:( Python or MATLAB)
(b) Butterworth filters
(c) Gaussian filters
package needed: numpy, opencv, matplotlib
"""

import cv2
import numpy as np
import matplotlib
#working in Pycharm community version
#so below 2 line adjusted to see my plot window one by one
matplotlib.use('TkAgg',force=True)
print("Switched to:",matplotlib.get_backend())
import matplotlib.pyplot as plt
from scipy import fftpack
from math import sqrt
from math import exp

#plot figure alignment
#plt.figure(figsize=(6.4*5, 4.8*5), constrained_layout=False)

# Load an image
img = cv2.imread("./humanface.jpg", 0)
# Perform Fourier Transform
original = np.fft.fft2(img)
#plt.imshow(img,"gray")

def apply_fourier_transform(img):
    # Apply 2D Fourier Transform
    f_transform = fftpack.fftshift(fftpack.fft2(img))
    magnitude_spectrum = np.log(np.abs(f_transform) + 1)
    return magnitude_spectrum
magnitude_spectrum = apply_fourier_transform(img)
#plt.imshow(magnitude_spectrum)

# Function to calculate distance between two points
def distance(point1,point2):
    return sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

# Butterworth low-pass filter function
def butterworthLP(D0,imgShape,n):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = 1/(1+(distance((y,x),center)/D0)**(2*n))
    return base

# Butterworth high-pass filter function
def butterworthHP(D0,imgShape,n):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = 1-1/(1+(distance((y,x),center)/D0)**(2*n))
    return base

# Gaussian low-pass filter function
def gaussianLP(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = exp(((-distance((y,x),center)**2)/(2*(D0**2))))
    return base

# Gaussian high-pass filter function
def gaussianHP(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = 1 - exp(((-distance((y,x),center)**2)/(2*(D0**2))))
    return base

# Applying Gaussian low-pass filter on the image
#plt.title("Gaussian low-pass filter image processing")
#all commented subplot can be uncommented and showed for scientific understanding
#plt.subplot(161), plt.imshow(img, "gray"), plt.title("Original Image")
#plt.subplot(163), plt.imshow(np.log(1+np.abs(original)), "gray"), plt.title("Spectrum")
centerGaussian = np.fft.fftshift(original)
#plt.subplot(164), plt.imshow(np.log(1+np.abs(centerGaussian)), "gray"), plt.title("Centered Spectrum")
gaussianLPCenter = centerGaussian * gaussianLP(50,img.shape)
#plt.subplot(165), plt.imshow(np.log(1+np.abs(gaussianLPCenter)), "gray"), plt.title("Centered Spectrum multiply Low Pass Filter")
LowPassGaussian = np.fft.ifftshift(gaussianLPCenter)
#plt.subplot(166), plt.imshow(np.log(1+np.abs(LowPassGaussian)), "gray"), plt.title("Decentralize")
inverse_gaussianLP = np.fft.ifft2(LowPassGaussian)
#plt.subplot(162), plt.imshow(np.abs(inverse_gaussianLP), "gray"), plt.title("Processed Image")
#plt.show()

# Applying Butterworth low-pass filter on the image
#plt.title("Butterworth low-pass filter image processing")
#all commented subplot can be uncommented and showed for scientific understanding
#plt.subplot(171), plt.imshow(img, "gray"), plt.title("Original Image")
#plt.subplot(173), plt.imshow(np.log(1+np.abs(original)), "gray"), plt.title("Spectrum")
centerButterworth = np.fft.fftshift(original)
#plt.subplot(174), plt.imshow(np.log(1+np.abs(centerButterworth)), "gray"), plt.title("Centered Spectrum")
butterworthLPCenter = centerButterworth * butterworthLP(50,img.shape,2)
#plt.subplot(175), plt.imshow(np.log(1+np.abs(butterworthLPCenter)), "gray"), plt.title("Centered Spectrum multiply Low Pass Filter")
LowPassButterworth = np.fft.ifftshift(butterworthLPCenter)
#plt.subplot(176), plt.imshow(np.log(1+np.abs(LowPassButterworth)), "gray"), plt.title("Decentralize")
inverse_butterworthLP = np.fft.ifft2(LowPassButterworth)
#plt.subplot(172), plt.imshow(np.abs(inverse_butterworthLP), "gray"), plt.title("Processed Image")
#plt.show()

# Applying Gaussian high-pass filter on the image
centerGaussian = np.fft.fftshift(original)
gaussianHPCenter = centerGaussian * gaussianHP(50,img.shape)
HighPassGaussian = np.fft.ifftshift(gaussianHPCenter)
inverse_gaussianHP = np.fft.ifft2(HighPassGaussian)

# Applying Butterworth high-pass filter on the image
centerButterworth = np.fft.fftshift(original)
butterworthHPCenter = centerButterworth * butterworthHP(50,img.shape,2)
HighPassButterworth = np.fft.ifftshift(butterworthHPCenter)
inverse_butterworthHP = np.fft.ifft2(HighPassButterworth)

#Display the Results
def display_images(img, magnitude_spectrum, inverse_butterworthLP, inverse_butterworthHP,
               inverse_gaussianLP, inverse_gaussianHP):
    plt.figure(figsize=(12, 8))
    plt.subplot(231), plt.imshow(img,"gray")
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(232), plt.imshow(magnitude_spectrum,"gray")
    plt.title('Magnitude Spectrum (Fourier)')
    plt.axis('off')
    plt.subplot(233), plt.imshow(np.abs(inverse_butterworthLP),"gray")
    plt.title('Butterworth Low-Pass')
    plt.axis('off')
    plt.subplot(234), plt.imshow(np.abs(inverse_butterworthHP),"gray")
    plt.title('Butterworth High-Pass')
    plt.axis('off')
    plt.subplot(235), plt.imshow(np.abs(inverse_gaussianLP),"gray")
    plt.title('Gaussian Low-Pass')
    plt.axis('off')
    plt.subplot(236), plt.imshow(np.abs(inverse_gaussianHP),"gray")
    plt.title('Gaussian High-Pass')
    plt.axis('off')
    plt.show()

# Display the results
display_images(img, magnitude_spectrum, inverse_butterworthLP, inverse_butterworthHP,
               inverse_gaussianLP, inverse_gaussianHP)