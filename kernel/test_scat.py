import numpy as np
import matplotlib.pyplot as plt
from scattering.morlet import *
from scattering.display import *
from scattering.scatmorlet import *
from parse import *


test = 1
if test == 0:
    n=5
    images = retrieve_first_train(n)
    plt.imshow(images[0])
    phi, psis = morlet_bank(32, sigma2=0.7, js=[1, 2, 3])
    for k in range(n):
        im1 = np.sum(images[k], 2)/3
        S1 = scattering(im1, phi, psis, order=2, sub_factor=1)
        plt.figure()
        plt.imshow(big_image(S1), interpolation="nearest")
        S2 = scattering(im1, phi, psis, order=2, sub_factor=8)
        plt.figure()
        plt.imshow(big_image(S2), interpolation="nearest")
    plt.show()
elif test == 1:
    images = testdb()
    images = np.sum(images, 3)/3.
    print("computing wavelet...")
    phi, psis = morlet_bank(32, sigma2 = 0.7, js=[1, 2, 3])
    print("scattering...")
    s_coefs = scatter_images(images, phi, psis)
    np.savetxt("test_scat_morlet_m2.csv", s_coefs, delimiter=",")
