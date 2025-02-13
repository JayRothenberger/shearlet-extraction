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

# do we need a switch for this?
from shearletNN.shearlets import getcomplexshearlets2D
from shearletNN.shearlet_utils import (
    frequency_shearlet_transform,
    shifted_frequency_shearlet_transform,
    spatial_shearlet_transform,
    ShearletTransformLoader,
)

from utils import setup, cleanup
from dahps import DistributedAsynchronousRandomSearch as DARS
from dahps.torch_utils import sync_parameters

from config import experiment_config as config

import torch
from torch import distributed as dist
import torchvision
from torchvision.transforms import v2
from torchvision import transforms

import gc
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

# TODO: replace with functions from the args
patch_size = 32
image_size = 64
batch_size_train = 128

rows, cols = image_size, image_size


def training_process(args, rank, world_size):

    # selector for the type of dataset we will train on.  We will need to download it to a special directory from the args to make use of lscratch
    ds_train, ds_val = select_dataset(args)

    sampler_train = torch.utils.data.distributed.DistributedSampler(ds_train, int(os.environ['WORLD_SIZE']), int(os.environ['RANK']), shuffle=True)

    train_loader = torch.utils.data.DataLoader(
        ds_train, batch_size=batch_size_train, shuffle=False, num_workers=0, sampler=sampler_train
    )

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

    shearlets = shearlets[:args.n_shearlets]


    def shearlet_transform(img):
        return symsqrt(hartley_shearlet_transform(img.to(0), shearlets.to(0), patch_size))


    train_loader = ShearletTransformLoader(train_loader, shearlet_transform)


    mean, cov = loader_mean_cov(tqdm(train_loader))

    norm = Normalizer(mean, cov)


    train_loader = torch.utils.data.DataLoader(
        ds_train, batch_size=512, shuffle=False, num_workers=0, drop_last=True
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

    # TODO: log this (move the logging function to loggers.py)
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

    # TODO: change this to the fourier coordinates times the shearlets
    def shearlet_transform(img):
        return hartley_shearlet_transform(img.to(0), shearlets.to(0), patch_size)

    train_loader = ShearletTransformLoader(train_loader, shearlet_transform)
    val_loader = ShearletTransformLoader(val_loader, shearlet_transform)

    # TODO: add to parameter spectral normalization (doesn't work with compile)
    model = select_model(args)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 240)

    print("training model...")
    # TODO: add to parameter patience
    model, acc = model_run(model, optimizer, scheduler, args.epochs, args.accumulate, train_loader, val_loader, args.patience, torch.nn.CrossEntropyLoss(), sampler_train)


def create_parser():
    parser = argparse.ArgumentParser(description="Shearlet NN")

    parser.add_argument(
        "--epochs", type=int, default=20, help="training epochs (default: %(default)s)"
    )
    parser.add_argument(
        "--path", type=str, default='./hp_test', help="path for the hyperparameter search data"
    )


    return parser


def main(args, rank, world_size):
    setup(rank, world_size)

    device = rank % torch.cuda.device_count()
    print(f'rank {rank} running on device {device} (of {torch.cuda.device_count()})')
    torch.cuda.set_device(device)

    agent = DARS.from_config(args.path, config)

    agent = sync_parameters(rank, agent)

    args = agent.to_namespace(agent.combination)

    states, metric = training_process(args, rank, world_size)

    if rank == 0:
        print('saving checkpoint')
        agent.save_checkpoint(states)
        agent.finish_combination(metric)

    print('cleanup')
    cleanup()

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    torch.multiprocessing.set_start_method("spawn")

    main(args, rank, world_size)
