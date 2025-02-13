from shearletNN.shearlets import getcomplexshearlets2D
from shearletNN.shearlet_utils import (
    frequency_shearlet_transform,
    spatial_shearlet_transform,
    ShearletTransformLoader,
    shifted_frequency_shearlet_transform,
    shifted_spatial_shearlet_transform,
    hartley_shearlet_transform,
)
from shearletNN.complex_resnet import (
    complex_resnet18,
    complex_resnet34,
    complex_resnet50,
)

from shearletNN.complex_deit import (
    vit_models,
    LPatchEmbed,
    Attention,
    Block,
    ComplexLayerNorm,
    complex_Lfreakformer_small_patch1_LS,
)

from shearletNN.layers import CReLU, ComplexLayerNorm
from functools import partial

from shearletNN.layers import CGELU, CReLU

import torch
import torchvision
from torchvision.transforms import v2
from torchvision import transforms

import gc
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse


patch_size = 32
image_size = 64
batch_size_train = 128

rows, cols = image_size, image_size


shearlets, shearletIdxs, RMS, dualFrameWeights = getcomplexshearlets2D(
    rows,
    cols,
    1,  # scales per octave
    3,  # shear level (something like O(log of directions))
    1,  # octaves
    0.5,  # alpha
    wavelet_eff_support=image_size,
    gaussian_eff_support=image_size,
)

shearlets = torch.tensor(shearlets).permute(2, 0, 1).type(torch.complex64).to(0)


def train(model, optimizer, loader, accumulate=1):
    model.train()
    loss = torch.nn.CrossEntropyLoss()

    for i, (X, y) in tqdm(enumerate(loader)):
        out = model(X.to(0))
        optimizer.zero_grad()
        l = loss(out, y.to(0)) / accumulate
        l.backward()
        if i % accumulate == (accumulate - 1):
            optimizer.step()


def accuracy(output, target, topk=(1,)):
    output = output.to(torch.device("cpu"))
    target = target.to(torch.device("cpu"))
    maxk = max(topk)
    batch_size = target.shape[0]

    _, idx = output.sort(dim=1, descending=True)
    pred = idx.narrow(1, 0, maxk).t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(dim=0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def epoch_accuracy(loader_s, student):
    student.eval()

    out_epoch_s = [
        accuracy(student(L.to(0)), y)[0].detach().cpu().item() for L, y in loader_s
    ]

    student.train()

    return sum(out_epoch_s) / len(out_epoch_s)


def test(network, test_loader):
    network.eval().to(0)
    test_loss = 0
    correct = 0
    total = 0
    test_losses = []
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data.to(0))
            test_loss += torch.nn.CrossEntropyLoss()(output, target.to(0)).item()
            pred = output.data.max(1, keepdim=True)[1].cpu()
            correct += pred.eq(target.data.view_as(pred)).sum()
            total += target.shape[0]
        test_loss /= total
        test_losses.append(test_loss)
        print(
            "\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss, correct, total, 100.0 * correct / total
            )
        )


class IndexSubsetDataset:
    def __init__(self, ds, inds):
        self.ds = ds
        self.inds = inds

    def __iter__(self):
        for i in range(len(self.inds)):
            yield self[i]

    def __getitem__(self, i):
        return self.ds[self.inds[i]]

    def __len__(self):
        return len(self.inds)


def linearleaves(module):
    # returns a list of pairs of (parent, submodule_name) pairs for all submodule leaves of the current module
    if isinstance(module, torch.nn.Linear):
        return [(module, None)]

    linear_children = []
    for name, mod in module.named_modules():
        if isinstance(mod, torch.nn.Linear):
            linear_children.append((name, module))
    return linear_children


def linearleaves(module):
    # returns a list of pairs of (parent, submodule_name) pairs for all submodule leaves of the current module
    if isinstance(module, torch.nn.Linear):
        return [(module, None)]

    linear_children = []
    for name, mod in module.named_modules():
        if isinstance(mod, torch.nn.Linear) or isinstance(mod, torch.nn.Conv2d):
            linear_children.append((name, module))
    return linear_children


def getattrrecur(mod, s):
    s = s.split(".")
    for substr in s:
        mod = getattr(mod, substr)
    return mod


def setattrrecur(mod, s, value):
    s = s.split(".")
    for substr in s[:-1]:
        mod = getattr(mod, substr)
    setattr(mod, s[-1], value)


def spectral_normalize(model):
    for name, mod in linearleaves(model):
        setattrrecur(
            model,
            name,
            torch.nn.utils.parametrizations.spectral_norm(getattrrecur(mod, name)),
        )

    return model


def batch_cov_3d(points, mean):
    """
    for our purposes we want to batch the covariance along the channel dimension (originally 1) and compute it over the batch dimension (originally 0)

    we need a covariance matrix for each channel along the batch dimension, so of shape (C, 2, 2)

    Input: points \in (B, C, H, W, D)

    """
    points = points - mean
    points = points.permute(1, 0, 2, 3, 4)  # Channels first for the reshape
    C, B, H, W, D = points.size()
    N = B * H * W
    diffs = points.reshape(C * N, D)
    prods = torch.bmm(diffs.unsqueeze(2), diffs.unsqueeze(1)).reshape(C, N, D, D)
    bcov = prods.sum(dim=1) / N
    return bcov  # (C, D, D)


def loader_mean_cov(loader):
    total = 0
    mean = None
    cov = None

    for x, y in loader:
        total += 1
        if mean is None:
            mean = torch.mean(x, dim=(0, 2, 3), keepdim=True).unsqueeze(-1)

        else:
            mean += torch.mean(x, dim=(0, 2, 3), keepdim=True).unsqueeze(-1)

        if cov is None:
            cov = batch_cov_3d(x.unsqueeze(-1), mean)

        else:
            cov = (((total - 2) / (total - 1)) * cov) + (
                (1 / total) * batch_cov_3d(x.unsqueeze(-1), mean / total)
            )

    return mean, cov


class Normalizer:
    def __init__(self, mean, cov, eps=1e-6):
        self.mean = mean
        self.cov = cov
        self.epsilon_matrix = torch.eye(1) * eps

        # Inv and sqrtm is done over 2 inner most dimension [..., M, M] so it should be [..., 2, 2] for us.
        # torch has no matrix square root, so we have
        L, Q = torch.linalg.eigh(
            torch.linalg.inv(
                (self.cov + self.epsilon_matrix.unsqueeze(0).to(self.cov.device)).to(
                    torch.float64
                )
            )
        )  # low precision dtypes not supported
        # eigenvalues of positive semi-definite matrices are always real (and non-negative)
        diag = torch.diag_embed(L ** (0.5))
        self.inv_sqrt_var = Q @ diag @ Q.mH  # var^(-1/2), (C, 2, 2)

    def __call__(self, inputs):
        # Separate real and imag so I go from shape [...] to [..., 2]
        inputs = inputs.unsqueeze(-1)

        zero_mean = inputs - self.mean
        # (C, 2, 2) @ (B, H, W, C, 2, 1) -> (B, H, W, C, 2, 1)
        inputs_hat = torch.matmul(
            self.inv_sqrt_var.to(inputs.dtype),
            zero_mean.permute(0, 2, 3, 1, 4).unsqueeze(-1),
        )
        # Then I squeeze to remove the last shape so I go from [..., 2, 1] to [..., 2].
        # Use reshape and not squeeze in case I have 1 channel for example.
        squeeze_inputs_hat = torch.reshape(
            inputs_hat, shape=inputs_hat.shape[:-1]
        ).permute(0, 3, 1, 2, 4)
        # Get complex data
        complex_inputs_hat = squeeze_inputs_hat[..., 0]

        return complex_inputs_hat


def main(args, rank, world_size):
    def repeat3(x):
        return x.repeat(3, 1, 1)[:3]


    train_transform = v2.Compose(
        [
            transforms.RandomResizedCrop((image_size, image_size), scale=(0.5, 1.0)),
            # transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            repeat3,
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = v2.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            repeat3,
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    ds_train = torchvision.datasets.Caltech101(
        "./", transform=train_transform, download=True
    )
    ds_train = IndexSubsetDataset(
        ds_train, sum([list(range(len(ds_train)))[i::5] for i in range(1, 5)], [])
    )

    ds_val = torchvision.datasets.Caltech101("./", transform=val_transform, download=True)
    ds_val = IndexSubsetDataset(ds_val, list(range(len(ds_val)))[0::5])


    ds_train = torchvision.datasets.CIFAR10(
        "../", transform=train_transform, download=True, train=True
    )

    ds_val = torchvision.datasets.CIFAR10(
        "../", transform=val_transform, download=True, train=False
    )


    train_loader = torch.utils.data.DataLoader(
        ds_train, batch_size=batch_size_train, shuffle=True, num_workers=0
    )

    shearlets = shearlets[:]


    def symlog(x, threshold=1):
        return torch.sign(x) * torch.log(1 + torch.abs(x / threshold))


    def symsqrt(x):
        return torch.sign(x) * torch.sqrt(torch.abs(x))


    def complex_symsqrt(x):
        real = x.real
        imag = x.imag

        return torch.complex(symsqrt(real), symsqrt(imag))


    def complex_symlog(x, threshold=1):
        real = x.real
        imag = x.imag

        return torch.complex(symlog(real, threshold), symlog(imag, threshold))


    def shearlet_transform(img):
        return symsqrt(hartley_shearlet_transform(img.to(0), shearlets.to(0), patch_size))


    train_loader = ShearletTransformLoader(train_loader, shearlet_transform)


    mean, cov = loader_mean_cov(tqdm(train_loader))

    norm = Normalizer(mean, cov)


    def repeat3(x):
        return x.repeat(3, 1, 1)[:3]


    train_transform = v2.Compose(
        [
            transforms.RandomResizedCrop((image_size, image_size), scale=(0.5, 1.0)),
            # transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            repeat3,
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = v2.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            repeat3,
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    ds_train = torchvision.datasets.Caltech101(
        "./", transform=train_transform, download=True
    )
    ds_train = IndexSubsetDataset(
        ds_train, sum([list(range(len(ds_train)))[i::5] for i in range(1, 5)], [])
    )

    ds_val = torchvision.datasets.Caltech101("./", transform=val_transform, download=True)
    ds_val = IndexSubsetDataset(ds_val, list(range(len(ds_val)))[0::5])


    ds_train = torchvision.datasets.CIFAR10(
        "../", transform=train_transform, download=True, train=True
    )

    ds_val = torchvision.datasets.CIFAR10(
        "../", transform=val_transform, download=True, train=False
    )


    train_loader = torch.utils.data.DataLoader(
        ds_train, batch_size=512, shuffle=False, num_workers=0
    )

    for x, y in tqdm(train_loader):
        plt.imshow(x[0].sum(0).real.cpu().numpy())
        plt.show()
        break

    shearlets = shearlets[:]


    def shearlet_transform(img):
        return norm(
            symsqrt(hartley_shearlet_transform(img.to(0), shearlets.to(0), patch_size))
        )


    train_loader = ShearletTransformLoader(train_loader, shearlet_transform)

    val_loader = torch.utils.data.DataLoader(
        ds_val, batch_size=batch_size_train, shuffle=False
    )

    val_loader = ShearletTransformLoader(val_loader, shearlet_transform)

    for x, y in tqdm(train_loader):
        assert list(x.shape) == [
            512,
            shearlets.shape[0] * 3,
            patch_size,
            patch_size,
        ], x.shape
        assert x.dtype == torch.float32, x.dtype
        print(x.real.mean((0, 2, 3)))

        print(x.real.std((0, 2, 3)))

        print(x.real.max())

        plt.imshow(x[0].sum(0).real.cpu().numpy())
        plt.show()
        plt.hist2d(
            x.flatten().real.cpu().numpy(),
            x.flatten().real.cpu().numpy(),
            bins=200,
            range=[[-10, 10], [-10, 10]],
        )
        plt.show()
        plt.hist(x.flatten().real.cpu().numpy(), bins=1023, range=[-5, 5])
        plt.show()

    train_loader = torch.utils.data.DataLoader(
        ds_train, batch_size=batch_size_train, shuffle=True, num_workers=0
    )

    val_loader = torch.utils.data.DataLoader(
        ds_val, batch_size=batch_size_train, shuffle=False
    )


    def shearlet_transform(img):
        return hartley_shearlet_transform(img.to(0), shearlets.to(0), patch_size)


    train_loader = ShearletTransformLoader(train_loader, shearlet_transform)
    val_loader = ShearletTransformLoader(val_loader, shearlet_transform)


    model = complex_resnet18(in_dim=shearlets.shape[0] * 3, complex=False)

    model = spectral_normalize(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 240)

    print("training model...")
    for epoch in range(240):
        print("epoch", epoch)
        train(model.to(0), optimizer, train_loader, accumulate=2)
        gc.collect()
        if epoch % 8 == 7 and epoch > 24:
            test(model, train_loader)
            test(model, val_loader)
        scheduler.step()


def create_parser():
    parser = argparse.ArgumentParser(description="MCT benchmark")

    parser.add_argument(
        "--patch_size", type=int, default=5, help="warmup epochs (default: 10)"
    )
    parser.add_argument(
        "--image_size", type=int, default=15, help="fpft epochs (default: 10)"
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="training epochs (default: 10)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=[4096, 8192, 16384],
        help="batch size for training (default: 64)",
    )
    parser.add_argument(
        "--patience", type=int, default=32, help="patience for training"
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=1024,
        help="test batch size for training (default: 64)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=[1e-3, 1e-4],
        help="learning rate for SGD (default 1e-3)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="IN1k",
        metavar="e",
        help="embeddings over which to compute the distances",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="/ourdisk/hpc/ai2es/jroth/AI2ES_DL_Torch/MCT/open_set_hp3",
        help="path for hparam search directory",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/ourdisk/hpc/ai2es/datasets/Imagenet/2012",
        help="path containing training dataset",
    )
    parser.add_argument(
        "--train_size", type=float, default=[0.1], help="size of the training set (%)"
    )
    parser.add_argument(
        "--balanced",
        type=bool,
        default=False,
        help="Balanced dataset subsetting if true, else stratified sampling",
    )
    parser.add_argument(
        "--approx",
        type=bool,
        default=[True, False],
        help="Perform the full-parameter finetuning step",
    )
    parser.add_argument(
        "--amp",
        type=bool,
        default=[True, False],
        help="Perform the full-parameter finetuning step",
    )

    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    torch.multiprocessing.set_start_method("spawn")

    main(args, rank, world_size)
