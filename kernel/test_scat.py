import numpy as np
import matplotlib.pyplot as plt
from scattering.morlet import *
from scattering.display import *
from scattering.scatmorlet import *
from parse import *
from colorspace import *


test = 1
if test == 0:
    n=1
    images = retrieve_first_train(n)
    plt.imshow(images[0])
    phi, psis, freqs = morlet_bank(32, sigma2=0.7, js=[1, 2, 3])
    for k in range(n):
        im1 = np.sum(images[k], 2)/3
        S1 = scattering_fr_decr(im1, phi, psis, freqs, order=2, sub_factor=1)
        plt.figure()
        plt.imshow(big_image(S1), interpolation="nearest")
        S2 = scattering(im1, phi, psis, order=2, sub_factor=1)
        plt.figure()
        plt.imshow(big_image(S2), interpolation="nearest")
    plt.show()
elif test == 1:
    images = testdb()
    images = rgb2opponent(images)
    #images = rgb2yuv(images)
    print("computing wavelets...")
    phi, psis, js = morlet_bank(32, sigma2 = 0.7, js=[1, 2, 3, 4])
    print("scattering...")
    s_coefs = scatter_color_images(images, phi, psis, orders=[1,2], sub_factor=16, js=js)
    np.savetxt("test_scat_m12_fredecOPP.csv", s_coefs, delimiter=",")
