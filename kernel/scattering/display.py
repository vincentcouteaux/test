import numpy as np

def big_image(scat):
    w_sc = scat[0].shape[1]
    h_sc = scat[0].shape[0]
    w=5*w_sc
    h=(len(scat)/5+1)*h_sc
    out = np.zeros((h, w))
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

