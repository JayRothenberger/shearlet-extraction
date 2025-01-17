from shearletNN.shearlets import getcomplexshearlets2D
from shearletNN.shearlet_utils import frequency_shearlet_transform, spatial_shearlet_transform, ShearletTransformLoader
# from shearletNN.complex_resnet import complex_resnet18, complex_resnet34, complex_resnet50
# from shearletNN.layers import CGELU

import torch
import torchvision
from torchvision.transforms import v2
from torchvision import transforms
from tqdm import tqdm
import os
import argparse

import gc

from shearletNN.deit import deit_small_patch16_LS

from dahps import DistributedAsynchronousRandomSearch, sync_parameters

### training utilities: TODO move to another file


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
    output = output.to(torch.device('cpu'))
    target = target.to(torch.device('cpu'))
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

    out_epoch_s = [accuracy(student(L.to(0)), y)[0].detach().cpu().item() for L, y in loader_s]

    student.train()

    return sum(out_epoch_s) / len(out_epoch_s)

def test(network, test_loader):
    network.eval().to(0)
    test_loss = 0
    correct = 0
    total = 0
    test_losses=[]
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data.to(0))
            test_loss += torch.nn.CrossEntropyLoss()(output, target.to(0)).item()
            pred = output.data.max(1, keepdim=True)[1].cpu()
            correct += pred.eq(target.data.view_as(pred)).sum()
            total += target.shape[0]
        test_loss /= total
        test_losses.append(test_loss)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, total,
        100. * correct / total))

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
    
### end training utilities, end TODO

def main(args):

    patch_size = 128
    image_size = 256
    batch_size_train = 64

    rows, cols = image_size, image_size


    shearlets, shearletIdxs, RMS, dualFrameWeights = getcomplexshearlets2D(	rows, 
                                                                            cols, 
                                                                            1, 
                                                                            3, 
                                                                            1, 
                                                                            0.5,
                                                                            wavelet_eff_support = image_size,
                                                                            gaussian_eff_support = image_size
                                                                            )

    shearlets = torch.tensor(shearlets).permute(2, 0, 1).type(torch.complex64).to(0)


    def repeat3(x):
        return x.repeat(3, 1, 1)[:3]

    transform = v2.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        repeat3,
    ])

    ds_train = torchvision.datasets.Caltech101('./', transform=transform, download = True)
    ds_train = IndexSubsetDataset(ds_train, sum([list(range(len(ds_train)))[i::5] for i in range(1, 5)], []))

    ds_val = torchvision.datasets.Caltech101('./', transform=transform, download = True)
    ds_val = IndexSubsetDataset(ds_val, list(range(len(ds_val)))[0::5])

    train_loader = torch.utils.data.DataLoader(
    ds_train,
    batch_size=batch_size_train, shuffle=True, num_workers=0)

    val_loader = torch.utils.data.DataLoader(
    ds_val,
    batch_size=batch_size_train, shuffle=False)

    for x, y in tqdm(train_loader):
        print(x.dtype)
        assert list(x.shape) == [batch_size_train, 3, image_size, image_size], x.shape
        break
    print('building model...')
    model = deit_small_patch16_LS(img_size=128, in_chans=3, num_classes=101)

    for p in model.parameters():
        assert not p.isnan().any()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # TODO: transform this into a model run function with a learning rate scheduler for the optimizer
    print('training model...')
    for epoch in range(32):
        print('epoch', epoch)
        train(model.to(0), optimizer, train_loader, accumulate=1)
        gc.collect()
        test(model, train_loader)
        test(model, val_loader)


def create_parser():
    parser = argparse.ArgumentParser(description='Shearlet Compression')
    
    parser.add_argument('--warmup_epochs', type=int, default=5, 
                        help='warmup epochs (default: 10)')
    parser.add_argument('--fpft_epochs', type=int, default=15, 
                        help='fpft epochs (default: 10)')
    parser.add_argument('--epochs', type=int, default=20, 
                        help='training epochs (default: 10)')
    parser.add_argument('-b', '--batch_size', type=int, default=[4096, 8192, 16384], 
                        help='batch size for training (default: 64)')
    parser.add_argument('-p', '--patience', type=int, default=32, 
                        help='patience for training')
    parser.add_argument('-tb', '--test_batch_size', type=int, default=1024,
                        help='test batch size for training (default: 64)')
    parser.add_argument('-lr', '--learning_rate', type=float, default=[1e-3, 1e-4],
                        help='learning rate for SGD (default 1e-3)')
    parser.add_argument('--dataset', type=str, default='IN1k', metavar='e',
                        help='embeddings over which to compute the distances')
    parser.add_argument('--path', type=str, default='/ourdisk/hpc/ai2es/jroth/AI2ES_DL_Torch/MCT/open_set_hp3', help='path for hparam search directory')
    parser.add_argument('--dataset_path', type=str, default='/ourdisk/hpc/ai2es/datasets/Imagenet/2012', help='path containing training dataset')
    parser.add_argument('--train_size', type=float, default=[0.1],
                        help='size of the training set (%)')
    parser.add_argument('--balanced', type=bool, default=False, 
                        help='Balanced dataset subsetting if true, else stratified sampling')
    parser.add_argument('--approx', type=bool, default=[True, False], 
                        help='Perform the full-parameter finetuning step')
    parser.add_argument('--amp', type=bool, default=[True, False], 
                        help='Perform the full-parameter finetuning step')
    
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    torch.multiprocessing.set_start_method('spawn')

    main(args, rank, world_size)
