import torch
import wandb
import matplotlib.pyplot as plt
from tqdm import tqdm
from thop import profile
import time

def wandb_init(args):
    wandb.init('ai2es', args.project, config=vars(args), name=f'{args.experiment_type} {args.dataset}')

def log_data_histograms(args, train_loader):
    for x, y in tqdm(train_loader):
        assert list(x.shape) == [
            args.batch_size,
            2 * 3 * args.n_shearlets if args.experiment_type == "shearlet" else 6,
            args.crop_size,
            args.crop_size,
        ], x.shape
        assert x.dtype == torch.float32, x.dtype

        plt.imshow(x[0].sum(0).cpu().numpy())
        plt.show()
        fig = plt.gcf()
        wandb.log({'sample image': wandb.Image(fig)})
        plt.hist(x[:, :x.shape[1] // 2].flatten().cpu().numpy(), bins=1023, range=[-5, 5])
        plt.show()
        fig = plt.gcf()
        wandb.log({'first component histogram (real or magnitude)': wandb.Image(fig)})
        plt.hist(x[:, x.shape[1] // 2:].flatten().cpu().numpy(), bins=1023, range=[-5, 5])
        plt.show()
        fig = plt.gcf()
        wandb.log({'second component histogram (imaginary or phase)': wandb.Image(fig)})
        var = torch.var(x, dim=(0, 1))
        plt.imshow(var > 0)
        fig = plt.gcf()
        wandb.log({'pixel-wise variance': wandb.Image(fig)})
        break

def log_flops_params(args, model, train_loader):
    model = model.to(torch.cuda.current_device())
    input = torch.randn(1, 2 * 3 * args.n_shearlets if args.experiment_type == "shearlet" else 6, args.crop_size, args.crop_size).to(torch.cuda.current_device())
    macs, params = profile(model, inputs=(input, ))
    wandb.log({'macs': macs, 'params': params})

def log_latency(args, model, train_loader):
    model = model.to(torch.cuda.current_device())
    input = torch.randn(1, 2 * 3 * args.n_shearlets if args.experiment_type == "shearlet" else 6, args.crop_size, args.crop_size).to(torch.cuda.current_device())
    start_time = time.perf_counter()
    result = model(input)
    end_time = time.perf_counter()
    latency = end_time - start_time
    wandb.log({'latency': latency})

def log_model_throughput(model, train_loader):
    total = 0
    start_time = time.perf_counter()
    for x, _ in tqdm(train_loader):
        with torch.no_grad():
            y = model(x)
            total += x.shape[0]
    end_time = time.perf_counter()
    wandb.log({'throughput': total / (end_time - start_time)})
