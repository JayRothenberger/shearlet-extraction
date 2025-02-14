import torch
import wandb
import gc
from tqdm import tqdm
import torch.distributed as dist
import os
from .config import data_dir, spec_dir, model_dir, pooling_dir
from torchvision import transforms
from torchvision.transforms import v2
from itertools import cycle
import glob
import time
from pathlib import Path
from argparse import Namespace

from shearletNN.shearlet_utils import ShearletTransformLoader
from shearletNN.shearlets import getcomplexshearlets2D

# function that trains the model
def model_run(
    model,
    optimizer,
    scheduler,
    epochs,
    accumulate,
    train_loader,
    val_loader,
    patience,
    loss_fn,
    sampler=None,
):
    best_val_acc = 0
    best_state = None
    epochs_since_improvement = 0

    if int(os.environ.get("WORLD_SIZE")) > 1:
        print(
            "because the world size is greater than 1 I assume you want to aggregate performance metrics across ranks"
        )
        print("please make sure each rank is running the training process")

    test_fn = test_ddp if int(os.environ.get("WORLD_SIZE")) > 1 else test
    train_fn = train_ddp if int(os.environ.get("WORLD_SIZE")) > 1 else train

    print("training model...")
    for epoch in range(epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        print("epoch", epoch)
        train_args = {
            "rank": int(os.environ.get("RANK"))
            if os.environ.get("RANK") is not None
            else None,
            "model": model.to(torch.cuda.current_device()),
            "loader": train_loader,
            "loss_fn": loss_fn,
            "optimizer": optimizer,
            "accumulate": accumulate,
        }
        train_fn(**train_args)
        gc.collect()
        test_args = {
            "rank": int(os.environ.get("RANK"))
            if os.environ.get("RANK") is not None
            else None,
            "model": model.to(torch.cuda.current_device()),
            "loader": val_loader,
            "loss_fn": loss_fn,
            "optimizer": optimizer,
            "accumulate": accumulate,
        }
        val_loss, val_acc = test_fn(**test_args)
        test_args = {
            "rank": int(os.environ.get("RANK"))
            if os.environ.get("RANK") is not None
            else None,
            "model": model.to(torch.cuda.current_device()),
            "loader": train_loader,
            "loss_fn": loss_fn,
            "optimizer": optimizer,
            "accumulate": accumulate,
        }
        train_loss, train_acc = test_fn(**test_args)

        wandb.log(
            {
                "val loss": val_loss,
                "val acc": val_acc,
                "train loss": train_loss,
                "train acc": train_acc,
            }
        )

        if val_acc > best_val_acc:
            epochs_since_improvement = 0
            best_val_acc = val_acc
            best_state = model.state_dict()
        else:
            if epochs_since_improvement >= patience:
                return best_state, best_val_acc
            epochs_since_improvement += 1

        scheduler.step(val_loss)

    return best_state, best_val_acc


## single-gpu local versions of model training functions
def train(model, loader, loss_fn, optimizer, accumulate=1, **kwargs):
    model.train()

    total = 0
    loss = 0

    for i, (X, y) in tqdm(enumerate(loader)):
        out = model(X.to(torch.cuda.current_device()))
        optimizer.zero_grad()
        l = loss_fn(out, y.to(torch.cuda.current_device())) / accumulate

        loss += l.detach()
        total += 1

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
        accuracy(student(L.to(torch.cuda.current_device())), y)[0].detach().cpu().item()
        for L, y in loader_s
    ]

    student.train()

    return sum(out_epoch_s) / len(out_epoch_s)


def test(model, loader, loss_fn, **kwargs):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    test_losses = []
    with torch.no_grad():
        for data, target in loader:
            output = model(data.to(torch.cuda.current_device()))
            test_loss += loss_fn(output, target.to(torch.cuda.current_device())).item()
            pred = output.data.max(1, keepdim=True)[1].cpu()
            correct += pred.eq(target.data.view_as(pred)).sum()
            total += target.shape[0]
        test_loss /= total
        test_losses.append(test_loss)

    return test_losses[-1], correct / total


### ddp versions of the functions required for model runs


def train_ddp(rank, model, loader, loss_fn, optimizer, accumulate=1, **kwargs):
    step = 0
    ddp_loss = torch.zeros(5).to(torch.cuda.current_device())
    model.train()

    for X, y in tqdm(loader):
        X, y = X.to(torch.cuda.current_device()), y.to(torch.cuda.current_device())
        output = model(X)  # ), update_precision=False)

        loss = loss_fn(output, y) / accumulate

        loss.backward()
        if step % accumulate == (accumulate - 1):
            optimizer.step()
            optimizer.zero_grad()

        ddp_loss[0] += loss.item()
        # we should be smarter about this when we have an ignore index:
        ddp_loss[1] += (
            torch.where(y != loss_fn.ignore_index, (output.argmax(1) == y), 0.0)
            .sum()
            .item()
        )
        ddp_loss[2] += torch.where(y != loss_fn.ignore_index, 1.0, 0.0).sum().item()
        ddp_loss[4] += 1

        step += 1

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    train_acc = ddp_loss[1] / ddp_loss[2]
    train_loss = ddp_loss[0] / ddp_loss[2]

    return train_acc, train_loss


def test_ddp(rank, model, loader, loss_fn, **kwargs):
    ddp_loss = torch.zeros(5).to(torch.cuda.current_device())
    model.eval()

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(torch.cuda.current_device()), y.to(torch.cuda.current_device())
            output = model(X, with_variance=False, update_precision=False)
            loss = loss_fn(output.type(torch.float32), y)
            ddp_loss[0] += loss.item()
            ddp_loss[1] += (
                torch.where(y != loss_fn.ignore_index, (output.argmax(1) == y), 0)
                .type(torch.float)
                .sum()
                .item()
            )

            ddp_loss[2] += y.numel()
            ddp_loss[4] += 1

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    test_acc = ddp_loss[1] / ddp_loss[2]
    test_loss = ddp_loss[0] / ddp_loss[2]
    test_jaccard = ddp_loss[3] / ddp_loss[4]

    return test_acc, test_jaccard, test_loss


### dataset utilities


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
        return int(len(self.ds) * self.reps)


class RepeatLoader:
    def __init__(self, loader, batches=1):
        self.loader = loader
        self.batches = batches

    def __iter__(self):
        for b, i in zip(cycle(self.loader), range(self.batches)):
            yield b

def move_data_to_lscratch(rank, args):
    start = time.time()

    if rank == 0:
        LSCRATCH = os.environ['LSCRATCH'] + '/'
        os.mkdir(f'{LSCRATCH}{spec_dir[args.dataset]["download_path"].split("/")[-1]}')
        os.system(f'scp -r {spec_dir[args.dataset]["download_path"]}/* {LSCRATCH}{spec_dir[args.dataset]["download_path"].split("/")[-1]}')

        print(glob.glob(f'{LSCRATCH}{spec_dir[args.dataset]["download_path"].split("/")[-1]}/*.tar.gz'))
        print(glob.glob(LSCRATCH))

        for path in glob.glob(f'{LSCRATCH}{spec_dir[args.dataset]["download_path"].split("/")[-1]}/*.tar.gz'):
            os.system(f'tar -xzf {path} -C {LSCRATCH}{spec_dir[args.dataset]["download_path"].split("/")[-1]}/')

        for path in glob.glob(f'{LSCRATCH}{spec_dir[args.dataset]["download_path"].split("/")[-1]}/*.tar'):
            os.system(f'tar -xf {path} -C {LSCRATCH}{spec_dir[args.dataset]["download_path"].split("/")[-1]}/')
        
        print(glob.glob(f'{LSCRATCH}{spec_dir[args.dataset]["download_path"].split("/")[-1]}/*'))
        print(glob.glob(f'{LSCRATCH}{spec_dir[args.dataset]["download_path"]}/*'))
        print(glob.glob(LSCRATCH))

        spec_dir[args.dataset]["download_path"] = f'{LSCRATCH}{spec_dir[args.dataset]["download_path"].split("/")[-1]}/'

        if args.dataset == 'food101':
            args.dataset_path = args.dataset_path + '/' + spec_dir[args.dataset]["download_path"].split("/")[-1]

        Path(os.path.join('/scratch/jroth/', f'done_{os.environ["SLURM_JOB_ID"]}')).touch()

    else:
        while not os.path.isfile(os.path.join('/scratch/jroth/', f'done_{os.environ["SLURM_JOB_ID"]}')):
            time.sleep(10)
        while os.path.getmtime(os.path.join('/scratch/jroth/', f'done_{os.environ["SLURM_JOB_ID"]}')) < (start + 10):
            time.sleep(10)
    
    print('data unzipped')


### data processing utility functions (move to the shearlets package probably)


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


### data preprocessing code


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


### model modification utility functions

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


def repeat3(x):
    return x.repeat(3, 1, 1)[:3]


def select_dataset(args):
    image_size = args.crop_size if vars(args).get('resize_to_crop') else args.image_size

    train_transform = v2.Compose(
        [
            transforms.RandomResizedCrop(
                (image_size, image_size), scale=(0.5, 1.0)
            ),
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
    # selector for the type of dataset we will train on.  We will need to download it to a special directory from the args to make use of lscratch
    if spec_dir[args.dataset]["train"] is not None and spec_dir[args.dataset]["test"]:
        ds_train = data_dir[args.dataset](
            spec_dir[args.dataset]["download_path"],
            transform=train_transform,
            download=False,
            **spec_dir[args.dataset]["train"],
        )

        ds_val = data_dir[args.dataset](
            spec_dir[args.dataset]["download_path"],
            transform=val_transform,
            download=False,
            **spec_dir[args.dataset]["test"],
        )
    else:
        ds_train = data_dir[args.dataset](
            spec_dir[args.dataset]["download_path"], transform=train_transform, download=False
        )
        ds_train = IndexSubsetDataset(
            ds_train, sum([list(range(len(ds_train)))[i::5] for i in range(1, 5)], [])
        )

        ds_val = data_dir[args.dataset](
            spec_dir[args.dataset]["download_path"], transform=val_transform, download=False
        )
        ds_val = IndexSubsetDataset(ds_val, list(range(len(ds_val)))[0::5])

    return ds_train, ds_val


def select_model(args):
    if args.model_type == "resnet":
        model_fn = model_dir[args.model]

        in_chans = 2 * 3 * args.n_shearlets if args.experiment_type == "shearlet" else 6
        in_chans = 3 if args.experiment_type == "baseline" else in_chans

        model_kwargs = {
            "img_size": args.crop_size,
            "in_dim": in_chans,
        }

        model = model_fn(**model_kwargs)

    elif args.model_type == "deit":
        model_fn = model_dir[args.model_type]
        in_chans = 2 * 3 * args.n_shearlets if args.experiment_type == "shearlet" else 6
        in_chans = 3 if args.experiment_type == "baseline" else in_chans
        deit_in_chans = in_chans if not args.conv_first else in_chans * 4

        assert isinstance(args.crop_size, int), type(args.crop_size)
        assert isinstance(args.patch_size, int), type(args.patch_size)

        model_kwargs = {
            "embed_dim": args.embed_dim,
            "num_heads": args.num_heads,
            "patch_size": args.patch_size,
            "img_size": args.crop_size,
            "in_chans": deit_in_chans,
        }

        torch.set_float32_matmul_precision("high")

        model = model_fn(**model_kwargs)

        if args.conv_first:
            model = torch.nn.Sequential(
                torch.nn.Conv2d(in_chans, deit_in_chans, 3, 1, "same"), model
            )
    else:
        raise NotImplementedError(f"unrecognized model type: {args.model_type}")

    if vars(args).get('spectral_normalize'):
        model = spectral_normalize(model)
    else:
        model = torch.compile(model)

    return model


class PoolingTransform:
    def __init__(
        self, fn, crop_size, norm, magphase=False, symlog=False, shearlets=None
    ):
        self.fn = fn
        self.crop_size = crop_size
        self.norm = norm
        self.magphase = magphase
        self.symlog = symlog
        self.shearlets = shearlets.to(torch.cuda.current_device()) if shearlets is not None else None

    def __call__(self, batch):
        img = self.fn(
            batch.to(torch.cuda.current_device()), self.shearlets, self.crop_size
        )

        if not torch.is_complex(img):
            return img

        if self.magphase:
            if self.symlog:
                return self.norm(torch.cat(to_symlog_magphase(img), 1))
            return self.norm(torch.cat(to_magphase(img), 1))
        else:
            if self.symlog:
                return self.norm(torch.cat(to_symlog_real_imag(img), 1))
            return self.norm(torch.cat(to_real_imag(img), 1))


def sync_picklable_object(rank, object):
    pass  # TODO: pull some of the code from dahps to do this


def to_magphase(img):
    phase = torch.angle(img) / torch.math.pi
    mag = torch.sqrt((img.real**2) + (img.imag**2))

    return mag, phase


def to_symlog_magphase(img):
    phase = torch.angle(img) / torch.math.pi
    mag = torch.sqrt((img.real**2) + (img.imag**2))

    return mag, phase


def to_real_imag(img):
    return img.real, img.imag


def to_symlog_real_imag(img):
    return symlog(img.real), symlog(img.imag)


def loader_min_max(train_loader):
    a_max = None
    a_min = None

    for x, y in train_loader:
        # this should be the max elementwise and pixelwise as different ones are likely to have different ranges
        a_max = torch.maximum(
            torch.max(x, dim=0)[0],
            a_max if a_max is not None else torch.max(x, dim=0)[0],
        )
        a_min = torch.minimum(
            torch.min(x, dim=0)[0],
            a_min if a_min is not None else torch.min(x, dim=0)[0],
        )

    return a_max.unsqueeze(0), a_min.unsqueeze(0)


class MinMaxNormalizer:
    def __init__(self, min, max):
        self.min = min.to(torch.cuda.current_device())
        self.max = max.to(torch.cuda.current_device())
        self.diff = self.max - self.min

    def __call__(self, batch):
        img = batch.to(torch.cuda.current_device())
        img = (2 * (img - self.min) / self.diff) - 1

        return img


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


def get_shearlets(args):
    rows, cols = args.image_size, args.image_size

    shearlets, shearletIdxs, RMS, dualFrameWeights = getcomplexshearlets2D(
        rows,
        cols,
        1,  # scales per octave
        3,  # shear level (something like O(log of directions))
        1,  # octaves
        0.5,  # alpha
        wavelet_eff_support=args.image_size,
        gaussian_eff_support=args.image_size,
    )
    shearlets = torch.tensor(shearlets).permute(2, 0, 1).type(torch.complex128).to(0)
    shearlets = shearlets[: args.n_shearlets]

    return shearlets


def select_transform(args, ds_train):
    shearlets = get_shearlets(args) if vars(args).get('n_shearlets') is not None else None

    train_loader = torch.utils.data.DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True, num_workers=0
    )

    def pooling_transform(img):
        img = pooling_dir[args.experiment_type](
            img.to(torch.cuda.current_device()),
            shearlets.to(torch.cuda.current_device()) if shearlets is not None else None,
            args.crop_size,
        )

        if not torch.is_complex(img):
            return img
        
        if vars(args).get('magphase'):
            if vars(args).get('symlog'):
                return torch.cat(to_symlog_magphase(img), 1)
            return torch.cat(to_magphase(img), 1)
        else:
            if vars(args).get('symlog'):
                return torch.cat(to_symlog_real_imag(img), 1)
            return torch.cat(to_real_imag(img), 1)

    train_loader = ShearletTransformLoader(train_loader, pooling_transform)

    if vars(args).get('channel_norm'):
        mean, cov = loader_mean_cov(tqdm(train_loader))
        norm = Normalizer(mean, cov)
    elif vars(args).get('pixel_norm'):
        a_max, a_min = loader_min_max(tqdm(train_loader))
        norm = MinMaxNormalizer(a_min, a_max)
    else:
        norm = torch.nn.Identity()

    return PoolingTransform(
        pooling_dir[args.experiment_type],
        args.crop_size,
        norm,
        vars(args).get('magphase') if vars(args).get('magphase') is not None else False,
        args.symlog if vars(args).get('symlog') else False,
        shearlets,
    )

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()