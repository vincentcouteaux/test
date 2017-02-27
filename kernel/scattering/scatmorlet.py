from morlet import *

def scatter_images(images, phi, psis, order=2, sub_factor=8):
    out = []
    c = 0
    for im in images:
        if c%100==0:
            print("scattering image #{}".format(c))
        c = c+1
        flat_l = []
        S = scattering(im, phi, psis, order, sub_factor)
        for s in S:
            flat_l.append(s.flatten())
        out.append(np.concatenate(flat_l))
    return np.stack(out)

