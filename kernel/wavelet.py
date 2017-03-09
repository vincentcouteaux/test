#import cv2
import numpy as np
import matplotlib.pyplot as plt

#def load_image( infilename ) :
#    img = image.open( infilename )
#    img.load()
#    data = np.asarray( img, dtype="int32" )
#    return data

def daubechies(n):
    if n == 1:
        return np.array([1, -1]), np.array([1, 1])
    elif n == 2:
        sc = [0.6830127, 1.1830127, 0.3169873, -0.1830127]
    elif n == 3:
        sc = [0.47046721, 1.14111692, 0.650365, -0.19093442, -0.12083221, 0.0498175]
    elif n == 4:
        sc = [0.32580343, 1.01094572, 0.89220014, -0.03957503, -0.26450717, 0.0436163, 0.0465036, -0.01498699]
    sc = np.array(sc)
    wt = (-1)**np.arange(sc.size)*np.flip(sc, 0)
    return wt, sc

def subsample(a, dim):
    if dim == 0:
        return a[2*np.arange(int(np.ceil(a.shape[0]/2.)))]
    else:
        return a[:, 2*np.arange(int(np.ceil(a.shape[1]/2.)))]

def wvt2D(Im, wt, sc):
    approx = np.zeros(Im.shape)
    details = np.zeros(Im.shape)
    for i, row in enumerate(Im):
        approx[i] = np.convolve(row, sc, mode="same")
        details[i] = np.convolve(row, wt, mode="same")
    approx = subsample(approx, 1)
    details = subsample(details, 1)
    #cat = np.concatenate((approx, details))
    approx_tot = np.zeros(approx.shape)
    details_v = np.zeros(approx.shape)
    details_d = np.zeros(details.shape)
    details_h = np.zeros(details.shape)
    for i in range(approx.shape[1]):
        approx_tot[:, i] = np.convolve(approx[:, i], sc, mode="same")
        details_v[:, i] = np.convolve(approx[:, i], wt, mode="same")
    for i in range(details.shape[1]):
        details_h[:, i] = np.convolve(details[:, i], sc, mode="same")
        details_d[:, i] = np.convolve(details[:, i], wt, mode="same")
    approx_tot = subsample(approx_tot, 0)
    details_v = subsample(details_v, 0)
    details_d = subsample(details_d, 0)
    details_h = subsample(details_h, 0)
    return approx_tot, details_v, details_d, details_h

def scat2D(Im, wt, sc, order, n_lin=np.abs):
    out = []
    a, dv, dd, dh = wvt2D(Im, wt, sc)
    out += [n_lin(a), n_lin(dv), n_lin(dd), n_lin(dh)]
    for r in range(order-1):
        new_out = []
        for im in out:
            a, dv, dd, dh = wvt2D(im, wt, sc)
            new_out += [n_lin(a), n_lin(dv), n_lin(dd), n_lin(dh)]
        out = new_out
    return out

def big_image(scat, w, h):
    out = np.zeros((w, h))
    w_sc = scat[0].shape[1]
    h_sc = scat[0].shape[0]
    wc = 0
    hc = 0
    for s in scat:
        s = (s - np.min(s))/(np.max(s) - np.min(s))
        out[hc:hc+h_sc, wc:wc+w_sc] = s
        wc = wc + w_sc
        if wc >= w:
            wc = 0
            hc = hc + h_sc
    return out

def concatScat(scat):
    flat_l = []
    for s in scat:
        flat_l.append(s.flatten())
    return np.concatenate(flat_l)

def scat_and_concat(images, wt, sc, order=3, n_lin=np.abs):
    out = []
    for i, im in enumerate(images):
        out.append(concatScat(scat2D(im, wt, sc, order, n_lin)))
    return np.stack(out)


if __name__ == "__main__":
    lena = cv2.imread("lena512.bmp")
    wt, sc = daubechies(4)
    lena = lena[:, :, 0]
    l = scat2D(lena, wt, sc, 1, lambda x: x * (x > 0))
    plt.imshow(big_image(l, 512, 512))
    #for i in l:
    #    plt.figure()
    #    plt.imshow(i)
    plt.show()
