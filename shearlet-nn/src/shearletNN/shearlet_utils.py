import torch
import numpy as np
import warnings
import math
import torchvision


def suppress_square(coeffs, i=1, crop=False):
    j = 1
    # coeffs[coeffs.shape[0] // 2 - j:coeffs.shape[0] // 2 + j,
    #        coeffs.shape[0] // 2 - j:coeffs.shape[0] // 2 + j] = 0
    # coeffs[coeffs.shape[0] // 2 - 1:coeffs.shape[0] // 2 + 1] = 0
    # coeffs[:, coeffs.shape[0] // 2 - 1:coeffs.shape[0] // 2 + 1] = 0

    if crop:
        coeffs = coeffs[..., i:-i, i:-i]
        # coeffs = torchvision.transforms.CenterCrop(4*i)(coeffs)
        # coeffs = torch.nn.functional.upsample(coeffs, (2*i, 2*i))
        # coeffs = torchvision.transforms.functional.resize(coeffs, (2*i, 2*i), torchvision.transforms.InterpolationMode.NEAREST_EXACT, antialias=False)
        """
        for j in range((coeffs.shape[-1] // 2) - 8, 0, -2):
            coeffs = torch.cat((
                coeffs[..., :-(j + 1)],
                coeffs[..., -j:],
            ), -1)

            coeffs = torch.cat((
                coeffs[..., :j],
                coeffs[..., (j + 1):]
            ), -1)
        
            coeffs = torch.cat((
                coeffs[..., :-(j + 1), :],
                coeffs[..., -j:, :],
            ), -2)

            coeffs = torch.cat((
                coeffs[..., :j, :],
                coeffs[..., (j + 1):, :]
            ), -2)
        """

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
    Xfreq = fftlib.fftshift(
        fftlib.fft2(fftlib.ifftshift(X, dim=(-2, -1))), dim=(-2, -1)
    )
    # print('frequency domain')
    # visual.complexImageShow(Xfreq / Xfreq.max())
    # plt.show()

    # compute shearlet coefficients at each scale
    # note that pointwise multiplication in the fourier domain equals
    # convolution in the time-domain
    Y = torch.zeros(
        (shearlets.shape[-2], shearlets.shape[-1]), dtype=torch.complex64
    ).to(0)

    Y = Xfreq.unsqueeze(0) * torch.conj(shearlets)

    Y = suppress_square(Y, (Y.shape[-1] - patch_size) // 2)

    # use torch so we can vectorize even this process
    coeffs = fftlib.fftshift(
        fftlib.ifft2(fftlib.ifftshift(Y, dim=(-2, -1))), dim=(-2, -1)
    )

    return coeffs


def batched_spatial_coefficients(X, shearlets, patch_size=32):
    fftlib = torch.fft

    shearlets = shearlets.type(torch.complex128).to(torch.cuda.current_device())
    X = X.to(torch.cuda.current_device())
    # get data in frequency domain
    Xfreq = fftlib.fftshift(
        fftlib.fft2(fftlib.ifftshift(X, dim=(-2, -1)).type(torch.complex128)),
        dim=(-2, -1),
    )
    # print('frequency domain')
    # visual.complexImageShow(Xfreq / Xfreq.max())
    # plt.show()

    # compute shearlet coefficients at each scale
    # note that pointwise multiplication in the fourier domain equals
    # convolution in the time-domain
    Y = Xfreq.unsqueeze(1) * torch.conj(shearlets).unsqueeze(0)

    if patch_size < X.shape[-1]:
        Y = suppress_square(Y, (Y.shape[-1] - patch_size) // 2, crop=True)

    # use torch so we can vectorize even this process
    coeffs = fftlib.fftshift(
        fftlib.ifft2(fftlib.ifftshift(Y, dim=(-2, -1))), dim=(-2, -1)
    )

    return coeffs


def batched_hartley_pooled(X, shearlets, patch_size=32):
    fftlib = torch.fft

    shearlets = shearlets.type(torch.complex128).to(torch.cuda.current_device())

    X = X.to(torch.cuda.current_device())
    # get data in frequency domain
    Xfreq = fftlib.fftshift(
        fftlib.fft2(fftlib.ifftshift(X, dim=(-2, -1)).type(torch.complex128)),
        dim=(-2, -1),
    )

    Xfreq = (Xfreq.real - Xfreq.imag).unsqueeze(1)

    if patch_size < X.shape[-1]:
        Y = suppress_square(Xfreq, (Xfreq.shape[-1] - patch_size) // 2, crop=True)

    # use torch so we can vectorize even this process
    coeffs = fftlib.fftshift(
        fftlib.ifft2(fftlib.ifftshift(Y, dim=(-2, -1))), dim=(-2, -1)
    )

    coeffs = coeffs.real - coeffs.imag
    n = coeffs.shape[-1] * coeffs.shape[-2]
    coeffs = coeffs / n

    return coeffs


def batched_fourier_pooled(X, shearlets, patch_size=32):
    fftlib = torch.fft
    X = X.to(torch.cuda.current_device())
    # get data in frequency domain
    Xfreq = fftlib.fftshift(
        fftlib.fft2(fftlib.ifftshift(X, dim=(-2, -1)).type(torch.complex128)),
        dim=(-2, -1),
    )
    # print('frequency domain')
    # visual.complexImageShow(Xfreq / Xfreq.max())
    # plt.show()

    # compute shearlet coefficients at each scale
    # note that pointwise multiplication in the fourier domain equals
    # convolution in the time-domain
    Y = Xfreq.unsqueeze(1)

    if patch_size < X.shape[-1]:
        Y = suppress_square(Y, (Y.shape[-1] - patch_size) // 2, crop=True)

    # use torch so we can vectorize even this process
    # coeffs = fftlib.fftshift(
    #     fftlib.ifft2(fftlib.ifftshift(Y, dim=(-2, -1))), dim=(-2, -1)
    # )

    return Y


def batched_image_fourier_pooled(X, shearlets, patch_size=32):
    fftlib = torch.fft
    X = X.to(torch.cuda.current_device())
    # get data in frequency domain
    Xfreq = fftlib.fftshift(
        fftlib.fft2(fftlib.ifftshift(X, dim=(-2, -1)).type(torch.complex128)),
        dim=(-2, -1),
    )
    # print('frequency domain')
    # visual.complexImageShow(Xfreq / Xfreq.max())
    # plt.show()

    # compute shearlet coefficients at each scale
    # note that pointwise multiplication in the fourier domain equals
    # convolution in the time-domain
    Y = Xfreq.unsqueeze(1)

    if patch_size < X.shape[-1]:
        Y = suppress_square(Y, (Y.shape[-1] - patch_size) // 2, crop=True)

    # use torch so we can vectorize even this process
    coeffs = fftlib.fftshift(
        fftlib.ifft2(fftlib.ifftshift(Y, dim=(-2, -1))), dim=(-2, -1)
    )

    return coeffs


def batched_shearlet_pooled(X, shearlets, patch_size=32):
    fftlib = torch.fft
    shearlets = shearlets.type(torch.complex128).to(torch.cuda.current_device())
    X = X.to(torch.cuda.current_device())
    # get data in frequency domain
    Xfreq = fftlib.fftshift(
        fftlib.fft2(fftlib.ifftshift(X, dim=(-2, -1)).type(torch.complex128)),
        dim=(-2, -1),
    )
    # print('frequency domain')
    # visual.complexImageShow(Xfreq / Xfreq.max())
    # plt.show()

    # compute shearlet coefficients at each scale
    # note that pointwise multiplication in the fourier domain equals
    # convolution in the time-domain
    Y = Xfreq.unsqueeze(1) * shearlets.unsqueeze(0)

    if patch_size < X.shape[-1]:
        Y = suppress_square(Y, (Y.shape[-1] - patch_size) // 2, crop=True)

    # use torch so we can vectorize even this process
    # coeffs = fftlib.fftshift(
    #     fftlib.ifft2(fftlib.ifftshift(Y, dim=(-2, -1))), dim=(-2, -1)
    # )

    return Y


def frequency_coefficients(X, shearlets, patch_size=32):
    fftlib = torch.fft
    coeffs = np.zeros(shearlets.shape, dtype=complex)

    # get data in frequency domain
    Xfreq = fftlib.fftshift(
        fftlib.fft2(fftlib.ifftshift(X, dim=(-2, -1))), dim=(-2, -1)
    )
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
    shearlets = shearlets.type(torch.complex128).to(torch.cuda.current_device())
    X = X.to(torch.cuda.current_device())
    # get data in frequency domain
    Xfreq = fftlib.fftshift(
        fftlib.fft2(fftlib.ifftshift(X, dim=(-2, -1)).type(torch.complex128)),
        dim=(-2, -1),
    )
    # print('frequency domain')
    # visual.complexImageShow(Xfreq / Xfreq.max())
    # plt.show()

    # compute shearlet coefficients at each scale
    # note that pointwise multiplication in the fourier domain equals
    # convolution in the time-domain
    X = Xfreq.unsqueeze(1) * torch.conj(shearlets).unsqueeze(0)

    if patch_size < X.shape[-1]:
        X = suppress_square(X, (X.shape[-1] - patch_size) // 2, crop=True)

    return X


def DHT2d(x: torch.Tensor):
    """Compute the DHT for a sequence x of length n using the FFT."""
    X = torch.fft.fft2(x)
    X = X.real - X.imag

    return X


def iDHT2d(x: torch.Tensor):
    """Compute the IDHT for a sequence x of length n using the FFT.

    Since the DHT is involutory, IDHT(x) = 1/n DHT(H) = 1/n DHT(DHT(x))
    """
    n = x.shape[-1] * x.shape[-2]

    x = DHT2d(x)

    x = x / n

    return x


def flip_periodic(x: torch.Tensor):
    # x[..., 1:, 1:] = x[..., 1:, 1:].flip((-2, -1))
    x = x.flip((-2, -1))

    return x


def dht_conv(x: torch.Tensor, y: torch.Tensor):
    """Computes the DHT of the convolution of x and y, sequences of length n, using FFT.

    This is a straightforward implementation of the convolution theorem for the
    DHT. See https://en.wikipedia.org/wiki/Discrete_Hartley_transform#Properties

    """

    Xflip = flip_periodic(x)
    Yflip = flip_periodic(y)

    Yeven = 0.5 * (y + Yflip)
    Yodd = 0.5 * (y - Yflip)

    Z = x * Yeven + Xflip * Yodd

    return Z


def batched_hartley_coefficients(X, shearlets, patch_size=32):
    fftlib = torch.fft

    shearlets = shearlets.type(torch.complex128).to(torch.cuda.current_device())

    X = X.to(torch.cuda.current_device())
    # get data in frequency domain
    Xfreq = fftlib.fftshift(
        fftlib.fft2(fftlib.ifftshift(X, dim=(-2, -1)).type(torch.complex128)),
        dim=(-2, -1),
    )

    Xfreq = Xfreq.real - Xfreq.imag
    hshearlets = torch.conj(shearlets).unsqueeze(0)
    hshearlets = hshearlets.real - hshearlets.imag

    # Y = Xfreq.unsqueeze(1) * torch.conj(shearlets).unsqueeze(0)

    Y = dht_conv(Xfreq.unsqueeze(1), hshearlets)

    if patch_size < X.shape[-1]:
        Y = suppress_square(Y, (Y.shape[-1] - patch_size) // 2, crop=True)

    # use torch so we can vectorize even this process
    coeffs = fftlib.fftshift(
        fftlib.ifft2(fftlib.ifftshift(Y, dim=(-2, -1))), dim=(-2, -1)
    )

    coeffs = coeffs.real - coeffs.imag
    n = coeffs.shape[-1] * coeffs.shape[-2]
    coeffs = coeffs / n

    return coeffs


def shifted_batched_frequency_coefficients(X, shearlets, patch_size=32):
    fftlib = torch.fft
    coeffs = torch.zeros(shearlets.shape, dtype=torch.complex128)
    X = X.to(torch.cuda.current_device())

    Xfreq = X.to(torch.float64)

    Xfreq = fftlib.fft2(Xfreq)

    Y = torch.zeros(
        (shearlets.shape[-2], shearlets.shape[-1]), dtype=torch.complex128
    ).to(torch.cuda.current_device())

    Y = Xfreq.unsqueeze(1) * torch.conj(shearlets).unsqueeze(0) * shearlets.unsqueeze(0)

    coeffs = Y

    coeffs = fftlib.fftshift(coeffs, dim=(-2, -1))

    if patch_size < X.shape[-1]:
        coeffs = suppress_square(
            coeffs, (coeffs.shape[-1] - patch_size) // 2, crop=True
        )

    coeffs = fftlib.ifftshift(coeffs, dim=(-2, -1))

    return coeffs


def suppress_upper(img, size=32):
    img[..., -size:, -size:] = 0

    return img


def shifted_batched_spatial_coefficients(X, shearlets, patch_size=32):
    fftlib = torch.fft
    coeffs = torch.zeros(shearlets.shape, dtype=torch.complex128)
    X = fftlib.fftshift(X.to(torch.cuda.current_device()))
    shearlets = fftlib.fftshift(shearlets)

    Xfreq = X.to(torch.float64)

    Xfreq = fftlib.fft2(Xfreq)

    Y = torch.zeros(
        (shearlets.shape[-2], shearlets.shape[-1]), dtype=torch.complex128
    ).to(torch.cuda.current_device())

    Y = Xfreq.unsqueeze(1) * shearlets.unsqueeze(0)

    coeffs = Y

    if patch_size < X.shape[-1]:
        coeffs = suppress_square(
            coeffs, (coeffs.shape[-1] - patch_size) // 2, crop=True
        )

    # coeffs = suppress_upper(coeffs, patch_size // 2)

    coeffs = fftlib.ifft2(coeffs)

    return coeffs


def shifted_batched_frequency_reconstruction(coeffs, shearlets, dualFrameWeights):
    fftlib = torch.fft

    X = (fftlib.fft2(coeffs) * shearlets.unsqueeze(0)).sum(1)

    InversedualFrameWeights = 1 / dualFrameWeights
    InversedualFrameWeights = torch.nan_to_num(InversedualFrameWeights)

    X = fftlib.ifft2(X * fftlib.ifftshift(InversedualFrameWeights.unsqueeze(0)))

    return X.real


def batched_reconstruction(coeffs, shearlets, dualFrameWeights):
    fftlib = torch.fft
    # skipping useGPU stuff...
    X = (fftlib.fft2(coeffs) * shearlets.unsqueeze(0)).sum(1)

    InversedualFrameWeights = 1 / dualFrameWeights
    InversedualFrameWeights = torch.nan_to_num(InversedualFrameWeights)

    X = fftlib.ifft2((X * InversedualFrameWeights))

    return X.real


def spatial_shearlet_transform(img, shearlets, patch_size=32):
    img = torch.cat(
        [
            batched_spatial_coefficients(img[:, 0], shearlets, patch_size),
            batched_spatial_coefficients(img[:, 1], shearlets, patch_size),
            batched_spatial_coefficients(img[:, 2], shearlets, patch_size),
        ],
        1,
    ).type(torch.complex64)
    return img


def hartley_shearlet_transform(img, shearlets, patch_size=32):
    img = torch.cat(
        [
            batched_hartley_coefficients(img[:, 0], shearlets, patch_size),
            batched_hartley_coefficients(img[:, 1], shearlets, patch_size),
            batched_hartley_coefficients(img[:, 2], shearlets, patch_size),
        ],
        1,
    ).type(torch.float32)
    return img


def hartley_pooling_transform(img, shearlets, patch_size=32):
    img = torch.cat(
        [
            batched_hartley_pooled(img[:, 0], shearlets, patch_size),
            batched_hartley_pooled(img[:, 1], shearlets, patch_size),
            batched_hartley_pooled(img[:, 2], shearlets, patch_size),
        ],
        1,
    ).type(torch.float32)
    return img


def fourier_pooling_transform(img, shearlets, patch_size=32):
    img = torch.cat(
        [
            batched_fourier_pooled(img[:, 0], shearlets, patch_size),
            batched_fourier_pooled(img[:, 1], shearlets, patch_size),
            batched_fourier_pooled(img[:, 2], shearlets, patch_size),
        ],
        1,
    ).type(torch.complex64)
    return img


def image_fourier_pooling_transform(img, shearlets, patch_size=32):
    img = torch.cat(
        [
            batched_image_fourier_pooled(img[:, 0], shearlets, patch_size),
            batched_image_fourier_pooled(img[:, 1], shearlets, patch_size),
            batched_image_fourier_pooled(img[:, 2], shearlets, patch_size),
        ],
        1,
    ).type(torch.float32)
    return img


def shearlet_pooling_transform(img, shearlets, patch_size=32):
    img = torch.cat(
        [
            batched_shearlet_pooled(img[:, 0], shearlets, patch_size),
            batched_shearlet_pooled(img[:, 1], shearlets, patch_size),
            batched_shearlet_pooled(img[:, 2], shearlets, patch_size),
        ],
        1,
    ).type(torch.complex64)
    return img


def frequency_shearlet_transform(img, shearlets, patch_size=32):
    img = torch.cat(
        [
            batched_frequency_coefficients(img[:, 0], shearlets, patch_size),
            batched_frequency_coefficients(img[:, 1], shearlets, patch_size),
            batched_frequency_coefficients(img[:, 2], shearlets, patch_size),
        ],
        1,
    ).type(torch.complex64)
    return img


def shifted_frequency_shearlet_transform(img, shearlets, patch_size=32):
    img = torch.cat(
        [
            shifted_batched_frequency_coefficients(img[:, 0], shearlets, patch_size),
            shifted_batched_frequency_coefficients(img[:, 1], shearlets, patch_size),
            shifted_batched_frequency_coefficients(img[:, 2], shearlets, patch_size),
        ],
        1,
    ).type(torch.complex64)
    return img


def shifted_spatial_shearlet_transform(img, shearlets, patch_size=32):
    img = torch.cat(
        [
            shifted_batched_spatial_coefficients(img[:, 0], shearlets, patch_size),
            shifted_batched_spatial_coefficients(img[:, 1], shearlets, patch_size),
            shifted_batched_spatial_coefficients(img[:, 2], shearlets, patch_size),
        ],
        1,
    ).type(torch.complex64)
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
    return torch.complex(
        torch.abs(x),
        torch.nan_to_num(
            torch.arctan(x.imag / x.real),
            posinf=torch.math.pi / 2,
            neginf=-torch.math.pi / 2,
        ),
    )


def _complex_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

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
    tensor.real *= std * math.sqrt(2.0)
    tensor.imag *= std * math.sqrt(2.0)
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
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

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
    tensor.mul_(std * math.sqrt(2.0))
    tensor.add_(mean)

    # Clamp to ensure it's in the proper range
    tensor.clamp_(min=a, max=b)
    return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
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
