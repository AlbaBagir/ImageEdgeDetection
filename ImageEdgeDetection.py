import cv2
from cv2 import norm 
import numpy as np
import matplotlib.pyplot as plt
from numpy import array
from numpy.linalg import norm

img = cv2.imread('Flower.jpg') #read image
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #get original image
gray_image = cv2.imread('Flower.jpg',0) #

dx = cv2.Sobel(gray_image,cv2.CV_64F,1,0,3)
dy = cv2.Sobel(gray_image,cv2.CV_64F,0,1,3)

plt.subplot(1,4,1), plt.imshow(img), plt.title("Original image")
plt.subplot(1,4,2), plt.imshow(gray_image,cmap="gray"), plt.title("Gray Scale image")
plt.subplot(1,4,3), plt.imshow(dx,cmap="gray"), plt.title("dx")
plt.subplot(1,4,4), plt.imshow(dy,cmap="gray"), plt.title("dy")

dxNorm = array(dx)
dyNorm = array(dy)

l2dx = norm(dxNorm)
l2dy = norm(dyNorm)

print (l2dx, l2dy)

plt.show()