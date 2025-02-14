import torchvision
import ssl

from shearletNN.deit import vit_models

from shearletNN.complex_resnet import (
    complex_resnet18,
    complex_resnet34,
    complex_resnet50,
    complex_resnet101,
)

from shearletNN.shearlet_utils import (
    fourier_pooling_transform,
    image_fourier_pooling_transform,
    shearlet_pooling_transform,
)
### pooling function map from string names to pooling function classes

pooling_dir = {
    "fourier": fourier_pooling_transform,
    "baseline": image_fourier_pooling_transform,
    "shearlet": shearlet_pooling_transform,
}

### model registry utilities a map from string names to model classes

model_dir = {
    "deit": vit_models,
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
    "cifar10": torchvision.datasets.CIFAR10,
    "cifar100": torchvision.datasets.CIFAR100,
}

spec_dir = {
    "caltech101": {
        "train": None,
        "test": None,
        "classes": 101,
        "download_path": "/ourdisk/hpc/ai2es/datasets/caltech101",
    },
    "caltech256": {
        "train": None,
        "test": None,
        "classes": 257,
        "download_path": "/ourdisk/hpc/ai2es/datasets/caltech256",
    },
    "food101": {
        "train": {"split": "train"},
        "test": {"split": "test"},
        "classes": 101,
        "download_path": "/ourdisk/hpc/ai2es/datasets/Food101",
    },
    "inat2021": {
        "train": {"version": "2021_train"},
        "test": {"version": "2021_train"},
        "classes": 10_000,
        "download_path": "/ourdisk/hpc/ai2es/datasets/iNat2021",
    },
    "cifar10": {
        "train": {"train": True},
        "test": {"train": False},
        "classes": 10,
        "download_path": "./",  # "/ourdisk/hpc/ai2es/datasets/CIFAR100",
    },
    "cifar100": {
        "train": {"train": True},
        "test": {"train": False},
        "classes": 100,
        "download_path": "./",  # "/ourdisk/hpc/ai2es/datasets/CIFAR10",
    },
}

### Experimental Configuration Structure ###

# handles the low resolution datasets
low_res_config = {
    "key": "image_size",
    "values": [64, 128],
    "default": {"key": "crop_size", "values": [16, 32, 64], "default": None},
}
# handles the high resolution datasets
high_res_config = {
    "key": "image_size",
    "values": [128, 256],
    "default": {"key": "crop_size", "values": [16, 32, 64, 128], "default": None},
}

dataset_config = {
    "key": "pixel_norm",
    "values": [True, False],
    "default": {
        "key": "dataset",
        "values": [
            # "caltech101",
            # "caltech256",
            # "food101",
            # "inat2021",
            "cifar10",
            # "cifar100",
        ],
        "default": high_res_config,
        "cifar10": low_res_config,
        # "cifar100": low_res_config,
    },
    False: {
        "key": "channel_norm",
        "values": [True],
        "default": {
            "key": "dataset",
            "values": [
                # "caltech101",
                # "caltech256",
                # "food101",
                # "inat2021",
                "cifar10",
                # "cifar100",
            ],
            "default": high_res_config,
            "cifar10": low_res_config,
            # "cifar100": low_res_config,
        },
    },
}

optimization_config = {
    "key": "learning_rate",
    "values": [1e-3, 1e-4],
    "default": {
        "key": "batch_size",
        "values": [128],
        "default": {
            "key": "patience",
            "values": [25],
            "default": {
                "key": "spectral_norm",
                "values": [False],
                "default": {
                    "key": "accumulate",
                    "values": [2],
                    "default": dataset_config,
                },
            },
        },
    },
}

deit_config = {
    "key": "activation",
    "values": ["gelu"],
    "default": {
        "key": "patch_size",
        "values": [1, 2, 4, 8],  # patch size cannot be the same as image size
        "default": {
            "key": "num_heads",
            "values": [3, 6, 12],
            "default": {
                "key": "embed_dim",
                "values": [192, 384],
                "default": {
                    "key": "conv_first",
                    "values": [True, False],
                    "default": optimization_config,
                },
            },
        },
    },
}

resnet_config = {
    "key": "model",
    "values": [
        "complex_resnet18",
        "complex_resnet34",
        "complex_resnet50",
        "complex_resnet101",
    ],
    "default": optimization_config,
}

model_config = {
    "key": "model_type",
    "values": ["deit", "resnet"],
    "default": resnet_config,
    "deit": deit_config,
}

preprocessing_config = {
    "key": "magphase",
    "values": [True, False],
    "default": {"key": "symlog", "values": [True], "default": model_config},
}

experiment_config = {
    "root": {
        "key": "experiment_type",
        "values": ["shearlet", "fourier", "baseline"],
        "default": model_config,
        "baseline": {
            "key": "resize_to_crop",
            "values": [True, False],
            "default": model_config,
        },
        "shearlet": {
            "key": "n_shearlets",
            "values": [1, 3, 10],
            "default": preprocessing_config,
        },
        "fourier": preprocessing_config,
    },
    "check_unique": True,
    "repetitions": 1,
}

######
