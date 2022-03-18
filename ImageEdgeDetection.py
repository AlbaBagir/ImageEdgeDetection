import cv2
from cv2 import norm 
import numpy as np
import matplotlib.pyplot as plt
from numpy import array
from numpy.linalg import norm

gray_image = cv2.imread('Flower.jpg',0) #get gray_scale image

#read image as an array
img1 = array(gray_image)

#central difference formula f()
def centralDifference(image, h):
    img = image.astype(float)
    dx = (img[h:-h, 2*h:] - img[h:-h, 0:-2*h]) / (2*h)
    dy = (img[2*h:, h:-h] - img[0:-2*h, h:-h]) / (2*h) 
    img_lst = []
    img_lst.append(dx)
    img_lst.append(dy)
    norm = np.sqrt(dx**2 + dy**2)
    img_lst.append(norm)
    return img_lst

fig, img_lst = plt.subplots(1, 3) 
imgs = centralDifference(img1, 1)
img_lst[0].imshow(imgs[0], cmap="gray")
img_lst[1].imshow(imgs[1], cmap="gray")
img_lst[2].imshow(imgs[2], cmap="gray")
img_lst[0].set_title('dx')
img_lst[1].set_title('dy')
img_lst[2].set_title('norm')



plt.show()