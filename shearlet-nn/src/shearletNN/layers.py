import torch
from torch import nn
from typing import Any, Callable, List, Optional, Type, Union, Tuple
from torch.autograd import Function
import numpy as np


"""
TODO: support any number of dimensions for batch and layer norm, fix the dimension dependence in batch covariance
"""


class ComplexMaxPool2d(torch.nn.Module):
    """
    default behaviour interprets input as wavelet coefficients and pools by the aplitude
    """
    def __init__(self, kernel_size = 2, stride = 2, padding = 0, mode="absolute"):
        super(ComplexMaxPool2d, self).__init__()
        self.modes = ["magnitude", "conjugate", "absolute"]
        assert mode in self.modes
        self.mode = mode
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
    
    def forward(self, x):
        input_type = x.dtype
        x = x.type(torch.complex64) # gather not supported for complex half
        if self.mode == "magnitude":
            # take the maximum of the real part and use the corresponding imaginary part
            vals, inds = torch.nn.functional.max_pool2d(x.real, self.kernel_size, stride=self.stride, padding=self.padding, dilation=1, ceil_mode=False, return_indices=True)
            pooled = torch.gather(x.flatten(-2), -1, inds.flatten(-2)).reshape(vals.shape)
        elif self.mode == "absolute":
            # compute the max pool over the absolute value
            vals, inds = torch.nn.functional.max_pool2d(torch.abs(x).real, self.kernel_size, stride=self.stride, padding=self.padding, dilation=1, ceil_mode=False, return_indices=True)
            pooled = torch.gather(x.flatten(-2), -1, inds.flatten(-2)).reshape(vals.shape)
        elif self.mode == "conjugate":
            # compute the max pool over the complex conjugate
            vals, inds = torch.nn.functional.max_pool2d(torch.conj_physical(x).real, self.kernel_size, stride=self.stride, padding=self.padding, dilation=1, ceil_mode=False, return_indices=True)
            pooled = torch.gather(x.flatten(-2), -1, inds.flatten(-2)).reshape(vals.shape)

        return pooled.type(input_type)


class ComplexAdaptiveMaxPool2d(torch.nn.Module):
    """
    computes the adaptive max pool in the complex plane

    default behaviour interprets input as wavelet coefficients and pools by the aplitude
    """
    def __init__(self, output_size, mode="absolute"):
        super(ComplexAdaptiveMaxPool2d, self).__init__()
        self.modes = ["magnitude", "conjugate", "absolute"]
        assert mode in self.modes
        self.mode = mode
        self.output_size = output_size

    def forward(self, x):
        input_type = x.dtype
        # TODO: fix by separating into real and imaginary for gather (we can gather imaginary only and use the real result from the max pool op)
        x = x.type(torch.complex64) # gather not supported for complex half
        if self.mode == "magnitude":
            # take the maximum of the real part and use the corresponding imaginary part
            vals, inds = torch.nn.functional.adaptive_max_pool2d(x.real, self.output_size, return_indices=True)
            pooled = torch.gather(x.flatten(-2), -1, inds.flatten(-2)).reshape(vals.shape)
        elif self.mode == "absolute":
            # compute the max pool over the absolute value
            vals, inds = torch.nn.functional.adaptive_max_pool2d(torch.abs(x).real, self.output_size, return_indices=True)
            pooled = torch.gather(x.flatten(-2), -1, inds.flatten(-2)).reshape(vals.shape)
        elif self.mode == "conjugate":
            # compute the max pool over the complex conjugate
            vals, inds = torch.nn.functional.adaptive_max_pool2d(torch.conj_physical(x).real, self.output_size, return_indices=True)
            pooled = torch.gather(x.flatten(-2), -1, inds.flatten(-2)).reshape(vals.shape)

        return pooled.type(input_type)


def batch_cov(points):
    """
    for our purposes we want to batch the covariance along the channel dimension (originally 1) and compute it over the batch dimension (originally 0)

    we need a covariance matrix for each channel along the batch dimension, so of shape (C, 2, 2)

    Input: points \in (B, C, H, W, D)

    """
    points = points.permute(1, 0, 2, 3, 4) # Channels last for the reshape
    C, B, H, W, D = points.size()
    N = B * W * H
    mean = points.mean(dim=1).unsqueeze(1) # C, 1, H, W, D
    diffs = (points - mean).reshape(C * N, D)
    prods = torch.bmm(diffs.unsqueeze(2), diffs.unsqueeze(1)).reshape(C, N, D, D)
    bcov = prods.sum(dim=1) / (N - 1)  # Unbiased estimate
    return bcov  # (C, D, D)


def batch_cov_3d(points):
    """
    for our purposes we want to batch the covariance along the channel dimension (originally 1) and compute it over the batch dimension (originally 0)

    we need a covariance matrix for each channel along the batch dimension, so of shape (C, 2, 2)

    Input: points \in (B, C, H, W, D)

    """
    points = points.permute(2, 0, 1, 3) # Channels first for the reshape
    C, B, T, D = points.size()
    N = B * T
    mean = points.mean(dim=1).unsqueeze(1) # C, 1, H, W, D
    diffs = (points - mean).reshape(C * N, D)
    prods = torch.bmm(diffs.unsqueeze(2), diffs.unsqueeze(1)).reshape(C, N, D, D)
    bcov = prods.sum(dim=1) / (N - 1)  # Unbiased estimate
    return bcov  # (C, D, D)


class ComplexBatchNormalization(torch.nn.Module):
    """
    Complex Batch-Normalization as defined in section 3.5 of https://arxiv.org/abs/1705.09792
    """

    def __init__(self,      
                 num_features: int,            
                 eps: float = 0.001,
                 momentum: float = 0.99,
                 device: str = None,
                 dtype=torch.complex64,
                 dim: Union[List[int], Tuple[int], int] = 1, 
                 center: bool = True, 
                 scale: bool = True, 
                 beta_initializer=torch.zeros, 
                 gamma_initializer=torch.ones, 
                 moving_mean_initializer=torch.zeros, 
                 moving_variance_initializer=torch.ones, 
                 **kwargs):
        
        self.num_features = num_features
        self.my_dtype = dtype
        self.my_realtype = torch.zeros((1,), dtype=self.my_dtype).real.dtype
        self.eps = eps
        self.device = device

        if isinstance(dim, int):
            dim = [dim]
        self.dim = list(dim)

        for d in dim:
            assert d > -1 and isinstance(d, int), 'dim must be nonnegative integer'

        super(ComplexBatchNormalization, self).__init__(**kwargs)
        self.momentum = momentum
        self.beta_initializer = beta_initializer
        self.gamma_initializer = gamma_initializer
        self.moving_mean_initializer = moving_mean_initializer
        self.moving_variance_initializer = moving_variance_initializer
        self.center = center
        self.scale = scale

        factory_kwargs = {"device": device, "dtype": dtype}
        self.gamma_r = torch.nn.parameter.UninitializedParameter(**factory_kwargs)
        self.gamma_i = torch.nn.parameter.UninitializedParameter(**factory_kwargs)
        self.beta_r = torch.nn.parameter.UninitializedParameter(**factory_kwargs)
        self.beta_i = torch.nn.parameter.UninitializedParameter(**factory_kwargs)

        self.register_buffer("epsilon_matrix", torch.eye(2, dtype=self.my_realtype) * self.eps)

        desired_shape = [self.num_features]

        self.gamma_r = torch.nn.Parameter(
            data=self.gamma_initializer(size=tuple(desired_shape)),
            requires_grad=True
        )
        self.gamma_i = torch.nn.Parameter(
            data=torch.zeros(size=tuple(desired_shape)),
            requires_grad=True
        )  # I think I just need to scale with gamma, so by default I leave the imag part to zero
        self.beta_r = torch.nn.Parameter(
            data=self.beta_initializer(size=desired_shape),
            requires_grad=True
        )
        self.beta_i = torch.nn.Parameter(
            data=self.beta_initializer(size=desired_shape),
            requires_grad=True
        )
        # this is complex
        self.register_buffer("moving_mean", torch.complex(real=self.moving_mean_initializer(
                                                                          size=desired_shape,
                                                                          dtype=self.my_realtype
                                                                          ),
                                        imag=self.moving_mean_initializer(
                                                                          size=desired_shape,
                                                                          dtype=self.my_realtype
                                                                          )
                                                                          ))

        # this is always real because of how we computed it
        self.register_buffer("moving_var", torch.eye(2) * self.moving_variance_initializer(size=tuple(desired_shape) + (2, 2),
                                                                          dtype=self.my_realtype) / (2**(0.5)),)



    def forward(self, inputs):
        if self.training:
            # First get the mean and var
            mean = torch.complex(torch.mean(inputs.real, dim=(0, 2, 3), keepdim=True), 
                                 torch.mean(inputs.imag, dim=(0, 2, 3), keepdim=True)
                                 ).to(self.my_dtype) # 1, C, 1, 1
            var = batch_cov(torch.stack((inputs.real, inputs.imag), dim=-1)) # C, 2, 2

            # Now the train part with these values
            self.moving_mean = self.momentum * self.moving_mean + (1. - self.momentum) * mean[0].squeeze().detach()
            self.moving_var = self.moving_var * self.momentum + var.detach() * (1. - self.momentum)

            out = self._normalize(inputs, var, mean) # B, C, H, W
            # out = inputs
        else:
            out = self._normalize(inputs, self.moving_var, self.moving_mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
            # out = inputs

        if self.scale:
            gamma = torch.complex(self.gamma_r, self.gamma_i).type(self.my_dtype)
            out = gamma.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) * out

        if self.center:
            beta = torch.complex(self.beta_r, self.beta_i).type(self.my_dtype)
            out = out + beta.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        return out

    def _normalize(self, inputs, var, mean):
        """
        :inputs: Tensor
        :param var: Tensor of shape [..., 2, 2], if inputs dtype is real, var[slice] = [[var_slice, 0], [0, 0]]
        :param mean: Tensor with the mean in the corresponding dtype (same shape as inputs)
        """
        complex_zero_mean = inputs - mean
        # Inv and sqrtm is done over 2 inner most dimension [..., M, M] so it should be [..., 2, 2] for us.
        # torch has no matrix square root, so we have:

        L, Q = torch.linalg.eigh(torch.linalg.inv((var + self.epsilon_matrix.unsqueeze(0)).to(torch.float64))) # low precision dtypes not supported
        # eigenvalues of positive semi-definite matrices are always real (and non-negative)
        diag = torch.diag_embed(L ** (0.5))
        inv_sqrt_var = Q @ diag @ Q.mH # var^(-1/2), (C, 2, 2)

        # Separate real and imag so I go from shape [...] to [..., 2]
        zero_mean = torch.stack((complex_zero_mean.real, complex_zero_mean.imag), axis=-1).permute(0, 2, 3, 1, 4)
        # (C, 2, 2) @ (1, H, W, C, 2, 1) -> (1, H, W, C, 2, 1)
        inputs_hat = torch.matmul(inv_sqrt_var.to(self.my_realtype), zero_mean.unsqueeze(-1))
        # Then I squeeze to remove the last shape so I go from [..., 2, 1] to [..., 2].
        # Use reshape and not squeeze in case I have 1 channel for example.
        squeeze_inputs_hat = torch.reshape(inputs_hat, shape=inputs_hat.shape[:-1]).permute(0, 3, 1, 2, 4)
        # Get complex data
        complex_inputs_hat = torch.complex(squeeze_inputs_hat[..., 0], squeeze_inputs_hat[..., 1]).type(self.my_dtype)

        return complex_inputs_hat


class ComplexLayerNorm(torch.nn.Module):
    """
    x = torch.randn((50,20,100))

    layerNorm = torch.nn.LayerNorm(x.shape[-1], elementwise_affine = True)
    y1 = layerNorm(x)

    mean = torch.mean(x, dim=-1, keepdim=True)
    var = torch.square(x - mean).mean(dim=-1, keepdim=True)
    y2 = ((x - mean) / torch.sqrt(var + layerNorm.eps)) * layerNorm.weight + layerNorm.bias

    torch.allclose(y1, y2, atol=1e-5, rtol=1e-5)
    """
    def __init__(self,      
                 num_features: int,            
                 eps: float = 0.0001,
                 elementwise_affine=True,
                 device: str = None,
                 dtype=torch.complex64,
                 dim: Union[List[int], Tuple[int], int] = 1, 
                 beta_initializer=torch.zeros, 
                 gamma_initializer=torch.ones, 
                 **kwargs):
        
        self.num_features = num_features
        self.my_dtype = dtype
        self.my_realtype = torch.zeros((1,), dtype=self.my_dtype).real.dtype
        self.eps = eps
        self.device = device

        if isinstance(dim, int):
            dim = [dim]
        self.dim = list(dim)

        for d in dim:
            assert d > -1 and isinstance(d, int), 'dim must be nonnegative integer'

        super(ComplexLayerNorm, self).__init__(**kwargs)
        self.elementwise_affine = elementwise_affine
        self.beta_initializer = beta_initializer
        self.gamma_initializer = gamma_initializer

        factory_kwargs = {"device": device, "dtype": dtype}
        self.gamma_r = torch.nn.parameter.UninitializedParameter(**factory_kwargs)
        self.gamma_i = torch.nn.parameter.UninitializedParameter(**factory_kwargs)
        self.beta_r = torch.nn.parameter.UninitializedParameter(**factory_kwargs)
        self.beta_i = torch.nn.parameter.UninitializedParameter(**factory_kwargs)

        # self.epsilon_matrix = torch.nn.parameter.Buffer(data=torch.eye(2, dtype=self.my_realtype) * self.eps)

        self.register_buffer("epsilon_matrix", torch.eye(2, dtype=self.my_realtype) * self.eps)

        desired_shape = [self.num_features]

        self.gamma_r = torch.nn.Parameter(
            data=self.gamma_initializer(size=tuple(desired_shape)),
            requires_grad=True
        )
        self.gamma_i = torch.nn.Parameter(
            data=torch.zeros(size=tuple(desired_shape)),
            requires_grad=True
        )  # I think I just need to scale with gamma, so by default I leave the imag part to zero
        self.beta_r = torch.nn.Parameter(
            data=self.beta_initializer(size=desired_shape),
            requires_grad=True
        )
        self.beta_i = torch.nn.Parameter(
            data=self.beta_initializer(size=desired_shape),
            requires_grad=True
        )


    def forward(self, inputs):
        # First get the mean and var
        mean = torch.complex(torch.mean(inputs.real, dim=-1, keepdim=True), 
                                torch.mean(inputs.imag, dim=-1, keepdim=True)
                                ).to(self.my_dtype) # B, 1, 1
        
        var = batch_cov_3d(torch.stack((inputs.real, inputs.imag), dim=-1)) # C, 2, 2

        out = self._normalize(inputs, var, mean) # B, C, H, W
        # out = inputs

        if self.elementwise_affine:
            gamma = torch.complex(self.gamma_r, self.gamma_i).type(self.my_dtype)
            out = gamma.unsqueeze(0).unsqueeze(0) * out

            beta = torch.complex(self.beta_r, self.beta_i).type(self.my_dtype)
            out = out + beta.unsqueeze(0).unsqueeze(0)

        return out

    def _normalize(self, inputs, var, mean):
        """
        :inputs: Tensor
        :param var: Tensor of shape [..., 2, 2], if inputs dtype is real, var[slice] = [[var_slice, 0], [0, 0]]
        :param mean: Tensor with the mean in the corresponding dtype (same shape as inputs)
        """
        complex_zero_mean = inputs - mean
        # Inv and sqrtm is done over 2 inner most dimension [..., M, M] so it should be [..., 2, 2] for us.
        # torch has no matrix square root, so we have:

        L, Q = torch.linalg.eigh(torch.linalg.inv((var + self.epsilon_matrix.unsqueeze(0)).to(torch.float64))) # low precision dtypes not supported
        # eigenvalues of positive semi-definite matrices are always real (and non-negative)
        diag = torch.diag_embed(L ** (0.5))
        inv_sqrt_var = Q @ diag @ Q.mH # var^(-1/2), (C, 2, 2)

        # Separate real and imag so I go from shape [...] to [..., 2]
        zero_mean = torch.stack((complex_zero_mean.real, complex_zero_mean.imag), axis=-1)
        # (C, 2, 2) @ (N, T, C, 2, 1) -> (N, T, C, 2, 1)
        inputs_hat = torch.matmul(inv_sqrt_var.to(self.my_realtype), zero_mean.unsqueeze(-1))
        # Then I squeeze to remove the last shape so I go from [..., 2, 1] to [..., 2].
        # Use reshape and not squeeze in case I have 1 channel for example.
        squeeze_inputs_hat = torch.reshape(inputs_hat, shape=inputs_hat.shape[:-1])
        # Get complex data
        complex_inputs_hat = torch.complex(squeeze_inputs_hat[..., 0], squeeze_inputs_hat[..., 1]).type(self.my_dtype)

        return complex_inputs_hat


class ComplexDropout(torch.nn.Module):
    def __init__(self, p):
        super(ComplexDropout, self).__init__()
        self.p = p

    def forward(self, x):
        # work around unimplemented dropout for complex
        if self.training:
            if x.is_complex():
                mask = torch.nn.functional.dropout(torch.ones_like(x.real), self.p)
                return x * mask
            else:
                return torch.nn.functional.dropout(x, self.p)
        else:
            return x


def complex_conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """complex 3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
        dtype=torch.complex64
    )


def complex_conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """complex 1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, dtype=torch.complex64)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
        dtype=torch.float32
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, dtype=torch.float32)



class ComplexFlatten(torch.nn.Module):
    def forward(self, x):
        flat = x.flatten(1)
        return torch.cat((flat.real, flat.imag), -1)


class ComplexConcat(torch.nn.Module):
    def __init__(self, dim=1):
        super(ComplexConcat, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.cat((x.real, x.imag), self.dim)


class CReLU(torch.nn.Module):

    def __init__(self, **kwargs):
        super(CReLU, self).__init__()
        self.rReLU = torch.nn.ReLU(**kwargs)
        self.iReLU = torch.nn.ReLU(**kwargs)

    def forward(self, x):
        return torch.complex(self.rReLU(x.real), self.iReLU(x.imag))
    
class RReLU(torch.nn.Module):

    def __init__(self, **kwargs):
        super(RReLU, self).__init__()
        self.rReLU = torch.nn.ReLU(**kwargs)

    def forward(self, x):
        return torch.complex(self.rReLU(x.real), x.imag)

class SinReLU(torch.nn.Module):

    def __init__(self, **kwargs):
        super(SinReLU, self).__init__()
        self.rReLU = torch.nn.ReLU(**kwargs)
        self.iReLU = torch.nn.ReLU(**kwargs)

    def forward(self, x):
        return torch.complex(self.rReLU(x.real), torch.sin(x.imag))
    

class CGELU(torch.nn.Module):

    def __init__(self, **kwargs):
        super(CGELU, self).__init__()
        self.rGELU = torch.nn.GELU(**kwargs)
        self.iGELU = torch.nn.GELU(**kwargs)

    def forward(self, x):
        # assert not x.isnan().any(), 'nan in gelu input'
        out = torch.complex(self.rGELU(x.real), self.iGELU(x.imag))
        assert not out.isnan().any(), 'nan in gelu'
        return out
    

class SinGELU(torch.nn.Module):

    def __init__(self, **kwargs):
        super(SinGELU, self).__init__()
        self.GELU = torch.nn.GELU(**kwargs)

    def forward(self, x):
        # return self.GELU(x)
        return torch.cat((torch.sin(x[..., :x.shape[-1] // 2]), self.GELU(x[..., x.shape[-1] // 2:])), -1)