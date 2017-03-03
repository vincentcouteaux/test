from morlet import *

def scatter_images(images, phi, psis, order=2, sub_factor=8, js=None):
    out = []
    c = 0
    for im in images:
        if c%100==0:
            print("scattering image #{}".format(c))
        c = c+1
        flat_l = []
        if js == None:
            S = scattering(im, phi, psis, order, sub_factor)
        else:
            S = scattering_fr_decr(im, phi, psis, js, order, sub_factor)
        for s in S:
            flat_l.append(s.flatten())
        out.append(np.concatenate(flat_l))
    return np.stack(out)

def scatter_color_images(images, phi, psis, order=2, sub_factor=8, js=None):
    images_red = images[:, :, :, 0]
    images_green = images[:, :, :, 1]
    images_blue = images[:, :, :, 2]
    print("Scattering red")
    scat_red = scatter_images(images_red, phi, psis, order, sub_factor, js)
    print("Scattering green")
    scat_green = scatter_images(images_green, phi, psis, order, sub_factor, js)
    print("Scattering blue")
    scat_blue = scatter_images(images_blue, phi, psis, order, sub_factor, js)
    print("concatenating...")
    return np.concatenate((scat_red, scat_green, scat_blue), 1)

