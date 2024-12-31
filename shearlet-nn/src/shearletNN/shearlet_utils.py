import torch 
import numpy as np

def suppress_square(coeffs, i=1, crop=False):

    if crop:
        coeffs = coeffs[..., i:-i, i:-i]
    else:
        coeffs[..., :, -i:] = 0
        coeffs[..., :, :i] = 0
        coeffs[..., :i, :] = 0
        coeffs[..., -i:, :] = 0

    return coeffs 

def spatial_coefficients(X, shearlets, patch_size=32):
    fftlib = torch.fft
    coeffs = torch.zeros(shearlets.shape, dtype=torch.complex64)
    X = X.to(0)

    # get data in frequency domain
    Xfreq = fftlib.fftshift(fftlib.fft2(fftlib.ifftshift(X, dim=(-2, -1))), dim=(-2, -1))
    # print('frequency domain')
    # visual.complexImageShow(Xfreq / Xfreq.max())
    # plt.show()

    # compute shearlet coefficients at each scale
    # note that pointwise multiplication in the fourier domain equals
    # convolution in the time-domain
    Y = torch.zeros((shearlets.shape[-2], shearlets.shape[-1]), dtype=torch.complex64).to(0)

    Y = Xfreq.unsqueeze(0) * torch.conj(shearlets)

    Y = suppress_square(Y, (Y.shape[-1] - patch_size) // 2)

    # use torch so we can vectorize even this process
    coeffs = fftlib.fftshift(fftlib.ifft2(fftlib.ifftshift(Y, dim=(-2, -1))), dim=(-2, -1))

    return coeffs

def batched_spatial_coefficients(X, shearlets, patch_size=32):
    fftlib = torch.fft
    coeffs = torch.zeros(shearlets.shape, dtype=torch.complex64)
    X = X.to(0)

    # get data in frequency domain
    Xfreq = fftlib.fftshift(fftlib.fft2(fftlib.ifftshift(X, dim=(-2, -1))), dim=(-2, -1))
    # print('frequency domain')
    # visual.complexImageShow(Xfreq / Xfreq.max())
    # plt.show()

    # compute shearlet coefficients at each scale
    # note that pointwise multiplication in the fourier domain equals
    # convolution in the time-domain
    Y = torch.zeros((shearlets.shape[-2], shearlets.shape[-1]), dtype=torch.complex64).to(0)

    Y = Xfreq.unsqueeze(1) * torch.conj(shearlets).unsqueeze(0)

    Y = suppress_square(Y, (Y.shape[-1] - patch_size) // 2)

    # use torch so we can vectorize even this process
    coeffs = fftlib.fftshift(fftlib.ifft2(fftlib.ifftshift(Y, dim=(-2, -1))), dim=(-2, -1))

    return coeffs

def frequency_coefficients(X, shearlets, patch_size=32):
    fftlib = torch.fft
    coeffs = np.zeros(shearlets.shape, dtype=complex)

    # get data in frequency domain
    Xfreq = fftlib.fftshift(fftlib.fft2(fftlib.ifftshift(X, dim=(-2, -1))), dim=(-2, -1))
    # print('frequency domain')
    # visual.complexImageShow(Xfreq / Xfreq.max())
    # plt.show()

    # compute shearlet coefficients at each scale
    # note that pointwise multiplication in the fourier domain equals
    # convolution in the time-domain
    X = Xfreq[np.newaxis] * np.conj(shearlets) * shearlets

    X = suppress_square(X, (X.shape[-1] - patch_size) // 2)

    return X


def batched_frequency_coefficients(X, shearlets, patch_size=32):
    fftlib = torch.fft
    X = X.to(0)
    # get data in frequency domain
    Xfreq = fftlib.fftshift(fftlib.fft2(fftlib.ifftshift(X, dim=(-2, -1))), dim=(-2, -1))
    # print('frequency domain')
    # visual.complexImageShow(Xfreq / Xfreq.max())
    # plt.show()

    # compute shearlet coefficients at each scale
    # note that pointwise multiplication in the fourier domain equals
    # convolution in the time-domain
    X = Xfreq.unsqueeze(1) * (torch.conj(shearlets).unsqueeze(0) * shearlets.unsqueeze(0))

    X = suppress_square(X, (X.shape[-1] - patch_size) // 2, True)

    return X

def spatial_shearlet_transform(img, shearlets, patch_size=32):
    img = torch.cat([
                     batched_spatial_coefficients(img[:, 0], shearlets, patch_size),
                     batched_spatial_coefficients(img[:, 1], shearlets, patch_size),
                     batched_spatial_coefficients(img[:, 2], shearlets, patch_size),
                     ], 1).type(torch.complex64)
    return img

def frequency_shearlet_transform(img, shearlets, patch_size=32):
    img = torch.cat([
                     batched_frequency_coefficients(img[:, 0], shearlets, patch_size),
                     batched_frequency_coefficients(img[:, 1], shearlets, patch_size),
                     batched_frequency_coefficients(img[:, 2], shearlets, patch_size),
                     ], 1).type(torch.complex64)
    return img


class ShearletTransformLoader:
    def __init__(self, loader, transform):
        self.loader = loader
        self.transform = transform

    def __iter__(self):
        for x, y in self.loader:
            yield self.transform(x), y