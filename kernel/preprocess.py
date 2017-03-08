import numpy as np
import matplotlib.pyplot as plt
from scattering.morlet import *
from scattering.display import *
from scattering.scatmorlet import *
from parse import *
from colorspace import *

name = "m12_OPPcolors"

test_images = testdb()
test_images = rgb2opponent(test_images)
train_images = traindb()
train_images = rgb2opponent(train_images)
#images = rgb2yuv(images)
print("computing wavelets...")
phi, psis, js = morlet_bank(32, sigma2 = 0.7, js=[1, 2, 3, 4])

print("scattering train...")
train_s_coefs = scatter_color_images(train_images, phi, psis, orders=[1,2], sub_factor=16, js=js)

print("scattering test...")
test_s_coefs = scatter_color_images(test_images, phi, psis, orders=[1,2], sub_factor=16, js=js)

mean = np.mean(train_s_coefs, 0)
std = np.std(train_s_coefs, 0)

#train_norm = (train_s_coefs - mean)/std
#test_norm = (test_s_coefs - mean)/std

#np.savetxt("train_" + name + ".csv", train_norm, delimiter=",")
#np.savetxt("test_" + name + ".csv", test_norm, delimiter=",")
np.savetxt("train_" + name + ".csv", train_s_coefs, delimiter=",")
np.savetxt("test_" + name + ".csv", test_s_coefs, delimiter=",")
