import torch 
import numpy as np
import warnings
import math

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

    X = suppress_square(X, (X.shape[-1] - patch_size) // 2, crop=True)

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

    X = suppress_square(X, (X.shape[-1] - patch_size) // 2, crop=True)

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


def phasor_to_magnitude_phase(x):
    """
    return magnitude/phase representation 
    """
    return torch.complex(torch.abs(x), torch.nan_to_num(torch.arctan(x.imag / x.real), posinf=torch.math.pi / 2, neginf= -torch.math.pi / 2))


def _complex_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    # Values are generated by using a truncated uniform distribution and
    # then using the inverse CDF for the normal distribution.
    # Get upper and lower cdf values
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    # Uniformly fill tensor with values from [l, u], then translate to
    # [2l-1, 2u-1].
    tensor.real.uniform_(2 * l - 1, 2 * u - 1)
    tensor.imag.uniform_(2 * l - 1, 2 * u - 1)
    assert not tensor.isnan().any()

    # Use inverse cdf transform for normal distribution to get truncated
    # standard normal
    tensor.real.erfinv_()
    tensor.imag.erfinv_()
    assert not tensor.isnan().any()

    # Transform to proper mean, std
    # tensor.mul_(std * math.sqrt(2.))
    tensor.real *= std * math.sqrt(2.)
    tensor.imag *= std * math.sqrt(2.)
    assert not tensor.isnan().any()
    tensor.real += mean
    tensor.imag += mean
    assert not tensor.isnan().any()

    # Clamp to ensure it's in the proper range
    tensor.real.clamp_(min=a, max=b)
    tensor.imag.clamp_(min=a, max=b)
    assert not tensor.isnan().any()

    return tensor

def _trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    # Values are generated by using a truncated uniform distribution and
    # then using the inverse CDF for the normal distribution.
    # Get upper and lower cdf values
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    # Uniformly fill tensor with values from [l, u], then translate to
    # [2l-1, 2u-1].
    tensor.uniform_(2 * l - 1, 2 * u - 1)

    # Use inverse cdf transform for normal distribution to get truncated
    # standard normal
    tensor.erfinv_()

    # Transform to proper mean, std
    tensor.mul_(std * math.sqrt(2.))
    tensor.add_(mean)

    # Clamp to ensure it's in the proper range
    tensor.clamp_(min=a, max=b)
    return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    NOTE: this impl is similar to the PyTorch trunc_normal_, the bounds [a, b] are
    applied while sampling the normal with mean/std applied, therefore a, b args
    should be adjusted to match the range of mean, std args.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    with torch.no_grad():
        if tensor.is_complex():
            return _complex_trunc_normal_(tensor, mean, std, a, b)
        else:
            return _trunc_normal_(tensor, mean, std, a, b)
    