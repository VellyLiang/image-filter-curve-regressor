#!/usr/bin/env python

import sys
from PIL import Image
import numpy as np
from scipy.optimize import least_squares
np.set_printoptions(formatter={'float': '{: 0.4f}'.format})

DIM = int(sys.argv[1])
CHANNEL = int(sys.argv[2])

original_image = Image.open('instant_photo.png')
(width, height) = original_image.size
total_pixels = width * height
original_image_data = np.array(original_image.getdata()) / 255.0

edited_image = Image.open('instant_photo_edited.png')
edited_image_data = np.array(edited_image.getdata()) / 255.0

def channel_data(channel):
    return (original_image_data[:, channel], edited_image_data[:, channel])

# Attempt to fit each color channel to a cubic polynomial
# target polynomial function
def f(a, x):
    return np.array([ a_i * x ** i for i, a_i in enumerate(a)]).sum(axis=0)
# residual function
def r(a, x, y):
    return y - f(a, x)

# training data
a0 = np.ones(DIM)

x_train, y_train = channel_data(CHANNEL)
x_train.sort()
y_train.sort()

res = least_squares(r, a0, args=(x_train, y_train))
a_final = res.x
print a_final

import matplotlib.pyplot as plt
plt.plot(x_train, y_train, 'r')
plt.plot(x_train, f(a_final, x_train))
plt.show()
