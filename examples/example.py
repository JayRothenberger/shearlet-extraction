from shearletNN.shearlets import getcomplexshearlets2D
from shearletNN.shearlet_utils import (
    frequency_shearlet_transform,
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
    # "inat2021": torchvision.datasets.INaturalist,
}

# TODO add a map mapping from string names to dataset download locations

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

### training utilities: TODO move to another file


def model_run(
    model, optimizer, scheduler, epochs, accumulate, train_loader, val_loader, patience
):
    best_val_acc = 0
    best_state = None
    epochs_since_improvement = 0

    print("training model...")
    for epoch in range(epochs):
        print("epoch", epoch)
        train(model.to(torch.cuda.current_device()), optimizer, train_loader, accumulate=accumulate)
        gc.collect()
        test(model, train_loader)
        val_loss, val_acc = test(model, val_loader)
        train_loss, train_acc = test(model, train_loader)

        wandb.log({'val loss': val_loss, 
                   'val acc': val_acc,
                   'train loss': train_loss,
                   'train acc': train_acc
                   })

        if val_acc > best_val_acc:
            epochs_since_improvement = 0
            best_val_acc = val_acc
            best_state = model.state_dict()
        else:
            if epochs_since_improvement >= patience:
                return model
            epochs_since_improvement += 1

        scheduler.step(val_loss)

    return best_state, best_val_acc


def train(model, optimizer, loader, accumulate=1):
    model.train()
    loss = torch.nn.CrossEntropyLoss()

    for i, (X, y) in tqdm(enumerate(loader)):
        out = model(X.to(torch.cuda.current_device()))
        optimizer.zero_grad()
        l = loss(out, y.to(torch.cuda.current_device())) / accumulate
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
        accuracy(student(L.to(torch.cuda.current_device())), y)[0].detach().cpu().item() for L, y in loader_s
    ]

    student.train()

    return sum(out_epoch_s) / len(out_epoch_s)


def test(network, test_loader):
    network.eval()
    test_loss = 0
    correct = 0
    total = 0
    test_losses = []
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data.to(torch.cuda.current_device()))
            test_loss += torch.nn.CrossEntropyLoss()(output, target.to(torch.cuda.current_device())).item()
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

    return test_losses[-1], correct / total


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


class RepeatDataset:
    def __init__(self, ds, reps=1):
        self.ds = ds
        self.reps = reps

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, i):
        return self.ds[i % len(self.ds)]

    def __len__(self):
        return len(self.ds) * self.reps


def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


### end training utilities, end TODO


def training_process(args):
    # experimental setup logistics

    device = int(os.environ["RANK"]) % torch.cuda.device_count()
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # weights and biases initialization process

    wandb.init(
        project="Shearlet Image Processing",
        entity="ai2es",
        name=f"{rank} {device} {torch.cuda.current_device()} : {args.model_type}",
        config={"args": vars(args)},
    )

    start = time.time()

    if rank == 0:
        LSCRATCH = os.environ['LSCRATCH'] + '/'
        os.mkdir(f'{LSCRATCH}{spec_dir[args.dataset]["download_path"].split("/")[-1]}')
        os.system(f'rsync -aP {spec_dir[args.dataset]["download_path"]}/* {LSCRATCH}{spec_dir[args.dataset]["download_path"].split("/")[-1]}')

        print(glob.glob(f'{LSCRATCH}{spec_dir[args.dataset]["download_path"].split("/")[-1]}/*.tar.gz'))
        print(glob.glob(LSCRATCH))

        for path in glob.glob(f'{LSCRATCH}{spec_dir[args.dataset]["download_path"].split("/")[-1]}/*.tar.gz'):
            os.system(f'tar -xzf {path} -C {LSCRATCH}{spec_dir[args.dataset]["download_path"].split("/")[-1]}/')

        for path in glob.glob(f'{LSCRATCH}{spec_dir[args.dataset]["download_path"].split("/")[-1]}/*.tar'):
            os.system(f'tar -xf {path} -C {LSCRATCH}{spec_dir[args.dataset]["download_path"].split("/")[-1]}/')
        
        print(glob.glob(f'{LSCRATCH}{spec_dir[args.dataset]["download_path"].split("/")[-1]}/*'))
        print(glob.glob(f'{LSCRATCH}{spec_dir[args.dataset]["download_path"]}/*'))
        print(glob.glob(LSCRATCH))
        print(args.dataset_path)

        if args.dataset == 'food101':
            args.dataset = args.dataset + spec_dir[args.dataset]["download_path"].split("/")[-1]

    else:
        while not os.path.isfile('done'):
            time.sleep(10)
        while os.path.getmtime('done') < (start + 100):
            time.sleep(10)
    
    print('data unzipped')

    patch_size = args.patch_size
    image_size = args.image_size
    batch_size_train = args.batch_size

    rows, cols = image_size, image_size

    shearlets, shearletIdxs, RMS, dualFrameWeights = getcomplexshearlets2D(
        rows,
        cols,
        1,
        3,
        1,
        0.5,
        wavelet_eff_support=image_size,
        gaussian_eff_support=image_size,
    )

    # manage how much of the shearlet coeffs we use (there are 20 max in our current configuration)

    shearlets = (
        torch.tensor(shearlets)
        .permute(2, 0, 1)
        .type(torch.complex64)
        .to(torch.cuda.current_device())
    )
    shearlets = shearlets[: args.num_shearlet_channels]

    def repeat3(x):
        return x.repeat(3, 1, 1)[:3]

    train_transform = v2.Compose(
        [
            transforms.RandomResizedCrop((image_size, image_size), scale=(0.5, 1.0)),
            transforms.ToTensor(),
            repeat3,
        ]
    )

    val_transform = v2.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            repeat3,
        ]
    )

    # selector for the type of dataset we will train on.  We will need to download it to a special directory from the args to make use of lscratch
    if spec_dir[args.dataset]["train"] is not None and spec_dir[args.dataset]["test"]:
        ds_train = data_dir[args.dataset](
            args.dataset_path,
            transform=train_transform,
            download=False,
            **spec_dir[args.dataset]["train"],
        )

        ds_val = data_dir[args.dataset](
            args.dataset_path,
            transform=val_transform,
            download=False,
            **spec_dir[args.dataset]["test"],
        )
    else:
        ds_train = data_dir[args.dataset](
            args.dataset_path, transform=train_transform, download=False
        )
        ds_train = IndexSubsetDataset(
            ds_train, sum([list(range(len(ds_train)))[i::5] for i in range(1, 5)], [])
        )

        ds_val = data_dir[args.dataset](
            args.dataset_path, transform=val_transform, download=False
        )
        ds_val = IndexSubsetDataset(ds_val, list(range(len(ds_val)))[0::5])

    train_loader = torch.utils.data.DataLoader(
        RepeatDataset(ds_train, 4 * args.accumulate),
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=0,
    )

    val_loader = torch.utils.data.DataLoader(
        ds_val, batch_size=batch_size_train, shuffle=False
    )

    # a selector for the type of model which will add a shearlet loader if needed and select the correct model

    # shearlet input experiments
    def shearlet_transform(img):
        return frequency_shearlet_transform(img.to(torch.cuda.current_device()), shearlets.to(torch.cuda.current_device()), patch_size)

    train_loader = ShearletTransformLoader(train_loader, shearlet_transform)

    val_loader = ShearletTransformLoader(val_loader, shearlet_transform)

    for x, y in tqdm(train_loader):
        # TODO: need to determine the size of the input based on the type of model and loader
        assert list(x.shape) == [batch_size_train, 3 * shearlets.shape[0], patch_size, patch_size], x.shape
        break

    model = model_dir[args.model_type](img_size=patch_size, in_chans=3 * shearlets.shape[0], in_dim=3 * shearlets.shape[0], num_classes=spec_dir[args.dataset]["classes"])

    for p in model.parameters():
        assert not p.isnan().any()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    return model_run(
        model,
        optimizer,
        scheduler,
        args.epochs,
        args.accumulate,
        train_loader,
        val_loader,
        args.patience,
    )


def create_parser():
    parser = argparse.ArgumentParser(description="Shearlet Compression")

    parser.add_argument(
        "--epochs", type=int, default=1024, help="training epochs (default: 10)"
    )    
    parser.add_argument(
        "--seed", type=int, default=13, help="random seed for torch"
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=256,
        help="batch size for training (default: 64)",
    )
    parser.add_argument(
        "-nsh",
        "--num_shearlet_channels",
        type=int,
        default=[1, 3, 10, 20],
        help="number of shearlet channels to use",
    )
    parser.add_argument(
        "-ps",
        "--patch_size",
        type=int,
        default=[32, 64, 96],
        help="batch size for training (default: 64)",
    )
    parser.add_argument(
        "-i",
        "--image_size",
        type=int,
        default=[128, 256, 512],
        help="batch size for training (default: 64)",
    )
    parser.add_argument(
        "-p", "--patience", type=int, default=32, help="patience for training"
    )
    parser.add_argument(
        "-a", "--accumulate", type=int, default=[1, 2, 4], help="patience for training"
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=1e-4,
        help="learning rate for SGD (default 1e-3)",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="./shearlet_hp3",
        help="path for hparam search directory",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default='./',
        help="path containing training dataset",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=[data_name for data_name in data_dir],
        help="path containing training dataset",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=[model_name for model_name in model_dir],
        help="name of model class",
    )

    return parser


def main(args, rank, world_size):
    setup(rank, world_size)
    torch.cuda.set_device(int(os.environ["RANK"]) % torch.cuda.device_count())

    # 4 * 4 * 3 * 3 * 4 * 12
    # = 6912

    search_space = [
        "accumulate",
        "patch_size",
        "image_size",
        "model_type",
        "num_shearlet_channels",
        "dataset",
    ]

    agent = sync_parameters(
        args, rank, search_space, DistributedAsynchronousRandomSearch
    )

    args = agent.to_namespace(agent.combination)

    states, metric = training_process(args)

    if rank == 0:
        print("saving checkpoint")
        agent.save_checkpoint(states)

    agent.finish_combination(metric)

    print("cleanup")
    cleanup()


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    torch.multiprocessing.set_start_method("spawn")

    main(args, rank, world_size)
