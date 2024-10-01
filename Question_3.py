# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 10:31:29 2024
@author: purnendumaity@gmail.com
Explain what a Kuwahara filter is, and apply it to the image using either Python or
MATLAB to demonstrate its effect.
package needed: opencv, pykuwahara,matplotlib
"""

import cv2
from pykuwahara import kuwahara
import matplotlib
#working in Pycharm community version
#so below 2 line adjusted to see my plot window one by one
matplotlib.use('TkAgg',force=True)
print("Switched to:",matplotlib.get_backend())
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image1 = cv2.imread('./modelwoman.jpg')

#The Kuwahara filter is a non-linear smoothing filter used in image processing
#for adaptive noise reduction. It is able to apply smoothing on the image
#while preserving the edges
filt1 = kuwahara(image1, method='mean', radius=3)
filt2 = kuwahara(image1, method='gaussian', radius=3)    # default sigma: computed by OpenCV
cv2.imwrite('modelwoman-kfilt-mean.jpg', filt1)
cv2.imwrite('modelwoman-kfilt-gauss.jpg', filt2)
image2=cv2.imread('./modelwoman-kfilt-mean.jpg')
image3=cv2.imread('./modelwoman-kfilt-gauss.jpg')
plt.figure(figsize=(6.4*5, 4.8*5), constrained_layout=False)
plt.title("non-linear smoothing filter image processing")
plt.subplot(131),
plt.imshow(mpimg.imread('./modelwoman.jpg')), plt.title("Original Image")
plt.subplot(132),
#OpenCV represents RGB images as multi-dimensional NumPy arrays…but in reverse order!
#This means that images are actually represented in BGR order rather than RGB!
#so we have to use COLOR_BGR2RGB
plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)),
plt.title("Kuwahara-kfilt-mean-image")
plt.subplot(133),
plt.imshow(cv2.cvtColor(image3, cv2.COLOR_BGR2RGB)),
plt.title("Kuwahara-kfilt-gauss-image")
plt.axis("off")
plt.show()

#Another Application is applying artistic effects
image4 = cv2.imread('./rock_beach.jpg')
# Set radius according to the image dimensions and the desired effect
filt1 = kuwahara(image4, method='mean', radius=4)
# NOTE: with sigma >= radius, this is equivalent to using 'mean' method
# NOTE: with sigma << radius, the radius has no effect
filt2 = kuwahara(image4, method='gaussian', radius=4, sigma=1.5)
cv2.imwrite('rock_beach-kfilt-mean.jpg', filt1)
cv2.imwrite('rock_beach-kfilt-gauss.jpg', filt2)
image5=cv2.imread('./rock_beach-kfilt-mean.jpg')
image6=cv2.imread('./rock_beach-kfilt-gauss.jpg')
plt.figure(figsize=(6.4*5, 4.8*5), constrained_layout=False)
plt.title("Artistic Effect apply image processing")
plt.subplot(141),
plt.imshow(mpimg.imread('./rock_beach.jpg')), plt.title("Original Image")
plt.subplot(142),
#OpenCV represents RGB images as multidimensional NumPy arrays…but in reverse order
#This means that images are actually represented in BGR order rather than RGB
#so we have to use COLOR_BGR2RGB to revert back to original order
plt.imshow(cv2.cvtColor(image5, cv2.COLOR_BGR2RGB)),
plt.title("Kuwahara-kfilt-mean-image")
plt.subplot(143),
plt.imshow(cv2.cvtColor(image6, cv2.COLOR_BGR2RGB)),
plt.title("Kuwahara-kfilt-gauss-image")
plt.axis("off")
plt.show()