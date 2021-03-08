# -*- coding: utf-8 -*-
"""
load images and convert to current I_app
"""
import gv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from skimage import color

image_names = ["zero.jpg", "one.jpg", "two.jpg", "three.jpg", "four.jpg", "five.jpg",
               "six.jpg", "seven.jpg","eight.jpg", "nine.jpg"]

"""
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
"""

def load_images(images_dir):
    for name in image_names:
        img = mpimg.imread(os.path.join(images_dir, name))
        img_gray = color.rgb2gray(img)
        img_gray= 1 - img_gray # specific regions of the digit have higher values
        gv.images.append(img_gray)
    #return images

def convert_im_to_I():
    gv.images = [10*gv.images[i] for i in range(10)]

"""
load_images("../images")
plt.imshow(images[9])
plt.show()
convert_im_to_I()
plt.imshow(images[9])
plt.show()
"""