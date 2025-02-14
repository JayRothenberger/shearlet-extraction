from shearletNN.shearlet_utils import ShearletTransformLoader

from scripts.utils import setup, cleanup
from dahps import DistributedAsynchronousRandomSearch as DARS
from dahps.torch_utils import sync_parameters
from scripts.config import experiment_config as config

import torch
import os
import argparse

from scripts.utils import select_dataset, select_model, select_transform, model_run
from scripts.loggers import wandb_init


def training_process(args, rank, world_size):
    wandb_init(args)

    # selector for the type of dataset we will train on.  We will need to download it to a special directory from the args to make use of lscratch
    ds_train, ds_val = select_dataset(args)

    sampler_train = torch.utils.data.distributed.DistributedSampler(
        ds_train, int(os.environ["WORLD_SIZE"]), int(os.environ["RANK"]), shuffle=True
    )

    train_loader = torch.utils.data.DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        sampler=sampler_train,
    )

    shearlet_transform = select_transform(args, ds_train)

    train_loader = ShearletTransformLoader(train_loader, shearlet_transform)

    val_loader = torch.utils.data.DataLoader(
        ds_val, batch_size=args.batch_size, shuffle=False
    )

    val_loader = ShearletTransformLoader(val_loader, shearlet_transform)

    train_loader = torch.utils.data.DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True, num_workers=0
    )

    val_loader = torch.utils.data.DataLoader(
        ds_val, batch_size=args.batch_size, shuffle=False
    )
    model = select_model(args)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 240)

    print("training model...")

    model, acc = model_run(
        model,
        optimizer,
        scheduler,
        args.epochs,
        args.accumulate,
        train_loader,
        val_loader,
        args.patience,
        torch.nn.CrossEntropyLoss(),
        sampler_train,
    )


def create_parser():
    parser = argparse.ArgumentParser(description="Shearlet NN")

    parser.add_argument(
        "--epochs", type=int, default=128, help="training epochs (default: %(default)s)"
    )
    parser.add_argument(
        "--path",
        type=str,
        default="./hp_test",
        help="path for the hyperparameter search data",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="shearletnn test",
        help="path for the hyperparameter search data",
    )

    return parser


def main(args, rank, world_size):
    setup(rank, world_size)

    device = rank % torch.cuda.device_count()
    print(f"rank {rank} running on device {device} (of {torch.cuda.device_count()})")
    torch.cuda.set_device(device)

    agent = DARS.from_config(args.path, config)

    agent = sync_parameters(rank, agent)

    args = agent.to_namespace(agent.combination)

    states, metric = training_process(args, rank, world_size)

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
