import torch
import wandb
import matplotlib.pyplot as plt
from tqdm import tqdm
from thop import profile
import time

def wandb_init(args):
    wandb.init('ai2es', args.project, config=vars(args), name=f'{args.experiment_type} {args.dataset}')

def log_data_histograms(args, train_loader):
    channels = 2 * 3 * args.n_shearlets if args.experiment_type == "shearlet" else 6
    channels = 3 if args.experiment_type == "baseline" else channels
    
    for x, y in tqdm(train_loader):
        assert list(x.shape) == [
            args.batch_size,
            channels,
            args.crop_size,
            args.crop_size,
        ], x.shape
        assert x.dtype == torch.float32, x.dtype

        plt.imshow(x[0].sum(0).cpu().numpy())
        fig = plt.gcf()
        wandb.log({'sample image': wandb.Image(fig)})
        plt.clf()

        plt.hist(x[:, :x.shape[1] // 2].flatten().cpu().numpy(), bins=1023, range=[-5, 5])
        fig = plt.gcf()
        wandb.log({'first component histogram (real or magnitude)': wandb.Image(fig)})
        plt.clf()

        plt.hist(x[:, x.shape[1] // 2:].flatten().cpu().numpy(), bins=1023, range=[-5, 5])
        fig = plt.gcf()
        wandb.log({'second component histogram (imaginary or phase)': wandb.Image(fig)})
        plt.clf()

        var = torch.var(x, dim=(0, 1)).cpu().numpy()
        plt.imshow(var > 0, vmax=1, vmin=0)
        fig = plt.gcf()
        wandb.log({'pixel-wise variance': wandb.Image(fig)})
        plt.clf()

        break

def log_flops_params(args, model, train_loader):
    channels = 2 * 3 * args.n_shearlets if args.experiment_type == "shearlet" else 6
    channels = 3 if args.experiment_type == "baseline" else channels

    model = model.to(torch.cuda.current_device())
    input = torch.randn(1, channels, args.crop_size, args.crop_size).to(torch.cuda.current_device())
    macs, params = profile(model, inputs=(input, ))
    wandb.log({'macs': macs, 'params': params})

def log_latency(args, model, train_loader):
    channels = 2 * 3 * args.n_shearlets if args.experiment_type == "shearlet" else 6
    channels = 3 if args.experiment_type == "baseline" else channels

    model = model.to(torch.cuda.current_device())
    input = torch.randn(4, channels, args.crop_size, args.crop_size).to(torch.cuda.current_device())
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
            y = model(x.to(torch.cuda.current_device()))
            total += x.shape[0]
    end_time = time.perf_counter()
    wandb.log({'throughput': total / (end_time - start_time)})
