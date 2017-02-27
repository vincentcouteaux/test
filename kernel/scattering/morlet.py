import numpy as np
import cv2
import matplotlib.pyplot as plt
from display import big_image

def morlet2d(j, theta, N, ksi, sigma2):
    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    R = np.array([[costheta, -sintheta], [sintheta, costheta]])
    sine = np.zeros((N,N), dtype=complex)
    env = np.zeros((N, N))
    #rang = np.arange(-N/2, N/2)/N*2.
    rang = np.linspace(-1, 1, N)
    C2 = 0.5
    for col, ux in enumerate(rang):
        for row, uy in enumerate(rang):
            sqnorm = (2**(2*j))*(ux**2 + uy**2)
            u_rot = (2**j)*np.dot(R, [ux, uy])
            dot = u_rot[0]*ksi
            #out[row, col] = (np.exp(1j*dot) - C2)*np.exp(-sqnorm/(2*sigma2))
            sine[row, col] = np.exp(1j*dot)
            env[row, col] = np.exp(-sqnorm/(2*sigma2))
    C2 = np.sum(sine*env)/np.sum(env)
    out = (sine - C2)*env
    out = out/np.sum(np.abs(out)**2)
    return out

def gaussian2d(N, j, sigma2):
    rang = np.linspace(-1, 1, N)
    out = np.zeros((N, N))
    for col, ux in enumerate(rang):
        for row, uy in enumerate(rang):
            sqnorm = (2**(2*j))*(ux**2 + uy**2)
            #sqnorm = (ux**2 + uy**2)
            out[row, col] = np.exp(-sqnorm/(2*sigma2))
    return out

def morlet_bank(N, sigma2 = 0.85**2, n_angles=6, js=[1, 2, 3], ksi=3*np.pi/4):
    phi = gaussian2d(N, min(js), sigma2)
    psis = np.zeros((n_angles*len(js), N, N), dtype=complex)
    c = 0
    for j in js:
        for theta in np.arange(n_angles)*np.pi/n_angles:
            psis[c] = morlet2d(j, theta, N, ksi, sigma2)
            c += 1
    return phi, psis

def convfft2d(i1, i2):
    return np.fft.ifftshift(np.fft.ifft2(np.fft.fft2(i1)*np.fft.fft2(i2)))

def subsamp(a, factor):
    return a[::factor, ::factor]

def scattering(image, phi, psis, order=1, sub_factor=8):
    out = [image]
    for m in range(order):
        scatm = []
        for o in out:
            for psi in psis:
                scatm.append(np.abs(convfft2d(o, psi)))
                #plt.imshow(scatm[-1])
                #plt.show()
        out = scatm
    for i, o in enumerate(out):
        out[i] = np.abs(convfft2d(o, phi))
        out[i] = subsamp(out[i], sub_factor)
    return out


if __name__ == "__main__":
    #plt.imshow(gaussian2d(20, 0.85**2))
    lena = cv2.imread("../lena512.bmp")[:, :, 0]
    print("creating wavelets")
    phi, psis = morlet_bank(512, sigma2=0.7**2, js=[4, 5, 6])
    print("scattering")
    S = scattering(lena, phi, psis, 2)
    plt.imshow(big_image(S))
    plt.show()

