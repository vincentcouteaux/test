import numpy as np
from parse import *

def change_color_space(images, matrix):
    dim = images.shape
    out = np.zeros(dim)
    for i in range(dim[0]):
        for j in range(dim[1]):
            for k in range(dim[2]):
                out[i, j, k] = np.dot(matrix, images[i, j, k])
    return out

def rgb2opponent(images):
    return change_color_space(images,
            np.array([[1./np.sqrt(2), -1./np.sqrt(2),0.],
                    [1./np.sqrt(6), 1./np.sqrt(6), -2./np.sqrt(6)],
                    [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)]]))

def rgb2yuv(images):
    return change_color_space(images, np.array([[.299, .587, .114],
                                                [-.14713, -.28886, .436],
                                                [.615, -.51498, -.10001]]))

if __name__ == "__main__":
    images = retrieve_first_train(5)
    yuv = rgb2yuv(images)
    opp = rgb2opponent(images)
    for k in range(5):
        plt.figure()
        plt.imshow(images[k])
        plt.title("original")
        plt.figure()
        plt.imshow(yuv[k])
        plt.title("yuv")
        plt.figure()
        plt.imshow(opp[k])
        plt.title("opponent")
    plt.show()


