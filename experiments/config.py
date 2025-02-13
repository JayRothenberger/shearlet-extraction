from shearletNN.shearlets import getcomplexshearlets2D
from shearletNN.shearlet_utils import (
    frequency_shearlet_transform,
    shifted_frequency_shearlet_transform,
    spatial_shearlet_transform,
    ShearletTransformLoader,
)
# from shearletNN.complex_resnet import complex_resnet18, complex_resnet34, complex_resnet50
# from shearletNN.layers import CGELU

import torch
import numpy as np
import torch.distributed as dist
import torchvision
from torchvision.transforms import v2
from torchvision import transforms
from tqdm import tqdm
import os
import argparse
import random
import gc
import wandb
import ssl
import shutil
import time
import glob

from shearletNN.complex_deit import (
    complex_freakformer_small_patch1_LS,
    complex_freakformer_small_patch2_LS,
    complex_freakformer_small_patch4_LS,
    complex_freakformer_base_patch1_LS,
    complex_freakformer_base_patch2_LS,
    complex_freakformer_base_patch4_LS,
    complex_rope_mixed_ape_deit_base_patch16_LS,
    complex_rope_mixed_ape_deit_small_patch16_LS,
    complex_rope_mixed_deit_base_patch16_LS,
    complex_rope_mixed_deit_small_patch16_LS,
)

from shearletNN.layers import CGELU

from shearletNN.complex_resnet import (
    complex_resnet18,
    complex_resnet34,
    complex_resnet50,
    complex_resnet101,
)

from dahps import DistributedAsynchronousRandomSearch, sync_parameters

### model registry utilities a map from string names to model classes

model_dir = {
    "complex_freakformer_small_patch1_LS": complex_freakformer_small_patch1_LS,
    "complex_freakformer_small_patch2_LS": complex_freakformer_small_patch2_LS,
    "complex_freakformer_small_patch4_LS": complex_freakformer_small_patch4_LS,
    "complex_freakformer_base_patch1_LS": complex_freakformer_base_patch1_LS,
    "complex_freakformer_base_patch2_LS": complex_freakformer_base_patch2_LS,
    "complex_freakformer_base_patch4_LS": complex_freakformer_base_patch4_LS,
    "complex_rope_mixed_ape_deit_base_patch16_LS": complex_rope_mixed_ape_deit_base_patch16_LS,
    "complex_rope_mixed_ape_deit_small_patch16_LS": complex_rope_mixed_ape_deit_small_patch16_LS,
    "complex_rope_mixed_deit_base_patch16_LS": complex_rope_mixed_deit_base_patch16_LS,
    "complex_rope_mixed_deit_small_patch16_LS": complex_rope_mixed_deit_small_patch16_LS,
    "complex_resnet18": complex_resnet18,
    "complex_resnet34": complex_resnet34,
    "complex_resnet50": complex_resnet50,
    "complex_resnet101": complex_resnet101,
}

ssl._create_default_https_context = ssl._create_unverified_context

### model dataset utilities:

# build a map from string names to dataset classes

data_dir = {
    "caltech101": torchvision.datasets.Caltech101,
    "caltech256": torchvision.datasets.Caltech256,
    "food101": torchvision.datasets.Food101,
    "inat2021": torchvision.datasets.INaturalist,
}

spec_dir = {
    "caltech101": {"train": None, "test": None, "classes": 101, "download_path": "/ourdisk/hpc/ai2es/datasets/caltech101"},
    "caltech256": {"train": None, "test": None, "classes": 257, "download_path": "/ourdisk/hpc/ai2es/datasets/caltech256"},
    "food101": {
        "train": {"split": "train"}, 
        "test": {"split": "test"}, 
        "classes": 101, 
        "download_path": "/ourdisk/hpc/ai2es/datasets/Food101"},
    "inat2021": {
        "train": {"version": "2021_train"},
        "test": {"version": "2021_train"},
        "classes": 10_000,
        "download_path": "/ourdisk/hpc/ai2es/datasets/iNat2021",
    },
}
