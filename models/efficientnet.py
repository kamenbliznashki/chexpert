"""
Implementation of EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks https://arxiv.org/pdf/1905.11946.pdf
Ref: Official author implemntation in tensorflow https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet
"""

import os
import math
import time
import argparse
import pprint
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
import torchvision.transforms as T
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
from tqdm import tqdm


SCALING_PARAMS = {# (width_coefficient, depth_coefficient, resolution, dropout_rate)
                  'efficientnet-b0': (1.0, 1.0, 224, 0.2),
                  'efficientnet-b1': (1.0, 1.1, 240, 0.2),
                  'efficientnet-b2': (1.1, 1.2, 260, 0.3),
                  'efficientnet-b3': (1.2, 1.4, 300, 0.3),
                  'efficientnet-b4': (1.4, 1.8, 380, 0.4),
                  'efficientnet-b5': (1.6, 2.2, 456, 0.4),
                  'efficientnet-b6': (1.8, 2.6, 528, 0.5),
                  'efficientnet-b7': (2.0, 3.1, 600, 0.5)}


parser = argparse.ArgumentParser()
# model
parser.add_argument('model', choices=list(SCALING_PARAMS.keys()), default='efficientnet-b0')
# action
parser.add_argument('--train', action='store_true', help='Train model.')
parser.add_argument('--evaluate', action='store_true', help='Evaluate a single model.')
parser.add_argument('--seed', type=int, default=0, help='Random seed to use.')
parser.add_argument('--cuda', type=int, help='Which cuda device to use.')
parser.add_argument('--mini_data', action='store_true', help='Truncate dataset to first entry only.')
# paths
parser.add_argument('--data_dir', default='', help='Location of datasets.')
parser.add_argument('--output_dir', help='Path to experiment output, config, checkpoints, etc.')
parser.add_argument('--restore', type=str, help='Path to a single model checkpoint to restore.')
# training params
parser.add_argument('--batch_size', type=int, default=256, help='Dataloaders batch size.')
parser.add_argument('--n_epochs', type=int, default=1, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.016, help='Base learning rate when the batch size is 256.')
parser.add_argument('--lr_mode', type=str, default='exponential', help='Mode for lr scheduler, e.g. constant or exponential.')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay regularization.')
parser.add_argument('--step', type=int, default=0, help='Current step of training (number of minibatches processed).')
parser.add_argument('--log_interval', type=int, default=1, help='Interval of num batches to show loss statistics.')
parser.add_argument('--eval_interval', type=int, default=10, help='Interval of epochs to evaluate and save model.')


# --------------------
# Model components
# --------------------

class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

class Squeeze(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.squeeze()
        if x.dim() == 1:
            x = x.unsqueeze(0)  # keep a batch dim
        return x

class DropConnect(nn.Dropout3d):
    # implemented here via dropout3d (cf torch docs): randomly zero out entire channels
    #   (a channel is a 3D feature map, e.g. the j-th channel of the i-th sample in the batched input is
    #   a 3D tensor input[i,j]) where layer input is (N,C,D,H,W).
    #   Here -- if we unsqueeze input to (1,N,C,H,W) dropout3d effectively zeros out datapoints, which matches the
    #   implementation by the authors multiplying the inputs by a binary tensor of shape (batch_size,1,1,1)
    def forward(self, x):
        return F.dropout3d(x.unsqueeze(0), self.p, self.training, self.inplace).squeeze(0)

class PaddedConv2d(nn.Conv2d):
    def forward(self, x):
        # `same` padding (cf https://www.tensorflow.org/api_docs/python/tf/nn/convolution)
        h_in, w_in = x.shape[2:]
        h_out, w_out = math.ceil(h_in/self.stride[0]), math.ceil(w_in/self.stride[1])
        padding = (math.ceil(max((h_out - 1) * self.stride[0] - h_in + self.dilation[0]*(self.kernel_size[0] - 1) + 1, 0)/2),
                   math.ceil(max((w_out - 1) * self.stride[1] - h_in + self.dilation[1]*(self.kernel_size[1] - 1) + 1, 0)/2))

        if padding[0] > 0 or padding[1] > 0:
            x = F.pad(x, [padding[1], padding[1], padding[0], padding[0]])

        return super().forward(x)

class SELayer(nn.Sequential):
    """ Squeeze and excitation layer cf Squeeze-and-Excitation Networks https://arxiv.org/abs/1709.01507 figure 2/3"""
    def __init__(self, n_channels, se_reduce_channels):
        super().__init__(nn.AdaptiveAvgPool2d(1),
                         nn.Conv2d(n_channels, se_reduce_channels, kernel_size=1),
                         Swish(),
                         nn.Conv2d(se_reduce_channels, n_channels, kernel_size=1),
                         nn.Sigmoid())

    def forward(self, x):
        return x * super().forward(x)

class MBConvBlock(nn.Sequential):
    """ mobile inverted residual bottleneck cf MnasNet https://arxiv.org/abs/1807.11626 figure 7b """
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio, drop_connect_rate):
        expand_channels = int(in_channels * expand_ratio)
        se_reduce_channels = max(1, int(in_channels * se_ratio))

        modules = []
        # expansion conv
        if expand_ratio != 1:
            modules += [nn.Conv2d(in_channels, expand_channels, kernel_size=1, bias=False),
                        nn.BatchNorm2d(expand_channels),
                        Swish()]
        modules += [
                # depthwise conv
                PaddedConv2d(expand_channels, expand_channels, kernel_size, stride, groups=expand_channels, bias=False),
                nn.BatchNorm2d(expand_channels),
                Swish(),
                # squeeze and excitation
                SELayer(expand_channels, se_reduce_channels),
                # projection conv
                nn.Conv2d(expand_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)]

        # drop connect -- only apply drop connect if skip (skip connection (in_channles==out_channels and stride=1))
        if in_channels==out_channels and stride==1:
            modules += [DropConnect(drop_connect_rate)]

        super().__init__(*modules)

    def forward(self, x):
        out = super().forward(x)
        if out.shape == x.shape:  # skip connection (in_channles==out_channels and stride=1)
            return out + x
        return out

class MBConvBlockRepeat(nn.Sequential):
    """ repeats MBConvBlock """
    def __init__(self, n_repeats, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio, drop_connect_rate):
        self.n_repeats = n_repeats
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.se_ratio = se_ratio
        self.drop_connect_rate = drop_connect_rate

        modules = []
        for i in range(n_repeats):
            modules += [MBConvBlock(in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio,
                                   drop_connect_rate*i/n_repeats)]
            in_channels = out_channels
            stride = 1
        super().__init__(*modules)


# --------------------
# Model
# --------------------

class EfficientNetB0(nn.Module):
    """ efficientnet b0 cf https://arxiv.org/abs/1905.11946 table 1 """
    def __init__(self, n_classes, dropout_rate=0.2, drop_connect_rate=0.2, bn_eps=1e-3, bn_momentum=0.01):
        super().__init__()

        self.stem = nn.Sequential(
                PaddedConv2d(3, 32, kernel_size=3, stride=2, bias=False),
                nn.BatchNorm2d(32),
                Swish())

        self.blocks = nn.Sequential(
                # blocks init (n_repeat, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio)
                MBConvBlockRepeat(1,  32,  16, 3, 1, 1, 0.25, drop_connect_rate),
                MBConvBlockRepeat(2,  16,  24, 3, 2, 6, 0.25, drop_connect_rate),
                MBConvBlockRepeat(2,  24,  40, 5, 2, 6, 0.25, drop_connect_rate),
                MBConvBlockRepeat(3,  40,  80, 3, 2, 6, 0.25, drop_connect_rate),
                MBConvBlockRepeat(3,  80, 112, 5, 1, 6, 0.25, drop_connect_rate),
                MBConvBlockRepeat(4, 112, 192, 5, 2, 6, 0.25, drop_connect_rate),
                MBConvBlockRepeat(1, 192, 320, 3, 1, 6, 0.25, drop_connect_rate))

        self.head = nn.Sequential(
                nn.Conv2d(320, 1280, kernel_size=1, bias=False),
                nn.BatchNorm2d(1280),
                Swish(),
                nn.AdaptiveAvgPool2d(1),
                Squeeze(),
                nn.Dropout(dropout_rate),
                nn.Linear(1280, n_classes))

        # initialize
        self.bn_eps = bn_eps
        self.bn_momentum = bn_momentum
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = self.bn_eps
                m.momentum = self.bn_momentum
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='conv2d')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5), mode='fan_in', nonlinearity='linear')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


def construct_model(model_name, n_classes):
    assert model_name in SCALING_PARAMS.keys(), 'Invalid model name.'

    # helper fn
    def _round_filters(filters, width_coeff, depth_divisor):
        new_filters = max(depth_divisor, int(filters * width_coeff + depth_divisor /2) // depth_divisor * depth_divisor)
        if new_filters < 0.9 * (filters * width_coeff):
            new_filters += depth_divisor
        return int(new_filters)

    # scaling params
    width_coeff, depth_coeff, resolution, dropout_rate = SCALING_PARAMS[model_name]
    depth_divisor=8

    # base model
    model = EfficientNetB0(n_classes, dropout_rate=dropout_rate)

    # scale base model
    # 1. scale stem conv and batch norm
    new_out_channels = _round_filters(model.stem[0].out_channels, width_coeff, depth_divisor)
    model.stem[0].__init__(model.stem[0].in_channels, new_out_channels, model.stem[0].kernel_size, model.stem[0].stride, bias=model.stem[0].bias is not None)
    model.stem[1].__init__(new_out_channels)

    # 2. scale blocks
    for i, b in enumerate(model.blocks):
        new_in_channels = _round_filters(b.in_channels, width_coeff, depth_divisor)
        new_out_channels = _round_filters(b.out_channels, width_coeff, depth_divisor)
        new_n_repeats = int(math.ceil(depth_coeff * b.n_repeats))

        model.blocks[i] = MBConvBlockRepeat(new_n_repeats, new_in_channels, new_out_channels, b.kernel_size, b.stride,
                                            b.expand_ratio, b.se_ratio, b.drop_connect_rate)
    # 3. scale head input conv
    model.head[0].__init__(new_out_channels, model.head[0].out_channels, model.head[0].kernel_size, bias=model.head[0].bias is not None)

    # re-init model
    model.reset_parameters()

    # update model name
    model.__class__.__name__ = model_name

    return model


# --------------------
# Metrics
# --------------------

@torch.no_grad()
def accuracy(output, target, topk=(1,5)):
    _, pred = output.topk(max(topk), dim=1, largest=True, sorted=True)
    correct = pred.eq(target.view(-1,1).expand(-1, pred.shape[1]))
    return [correct[:,:k].float().sum(1).mean(0).item() for k in topk]

# --------------------
# Train and evaluate
# --------------------

def train_epoch(model, train_dataloader, loss_fn, optimizer, scheduler, writer, epoch, args):
    model.train()

    with tqdm(total=len(train_dataloader), desc='Step at start {}; Training epoch {}/{}'.format(args.step, epoch+1, args.n_epochs)) as pbar:
        for x, y in train_dataloader:
            args.step += 1

            x, y = x.to(args.device), y.to(args.device)

            outputs = model(x)
            loss = loss_fn(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            pbar.set_postfix(loss = '{:.4f}'.format(loss.item()))
            pbar.update()

            # record
            if args.step % args.log_interval == 0:
                writer.add_scalar('train_loss', loss.item(), args.step)
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], args.step)

@torch.no_grad()
def evaluate(model, dataloader, loss_fn, args):
    model.eval()

    losses, top1s, top5s = 0, 0, 0
    for x, y in tqdm(dataloader):
        x, y = x.to(args.device), y.to(args.device)

        outputs = model(x)

        loss = loss_fn(outputs, y).item()
        top1, top5 = accuracy(outputs, y, topk=(1,5))

        # update sums
        losses += loss * x.shape[0]
        top1s  += top1 * x.shape[0]
        top5s  += top5 * x.shape[0]

    return losses/len(dataloader.dataset), top1s/len(dataloader.dataset), top5s/len(dataloader.dataset)

def train_and_evaluate(model, train_dataloader, valid_dataloader, loss_fn, optimizer, scheduler, writer, args):
    for epoch in range(args.n_epochs):
        train_epoch(model, train_dataloader, loss_fn, optimizer, scheduler, writer, epoch, args)

        if (epoch + 1) % args.eval_interval == 0:
            print('Evaluating...', end='\r')
            loss, top1, top5 = evaluate(model, valid_dataloader, loss_fn, args)
            print('Evaluate @ step {}: loss {:.4f}; acc@1 {:.4f}; acc@5 {:.4f}'.format(args.step, loss, top1, top5))
            writer.add_scalar('eval_loss', loss, args.step)
            writer.add_scalar('acc@top1', top1, args.step)
            writer.add_scalar('acc@top5', top5, args.step)

            # save model
            torch.save({'global_step': args.step,
                        'state_dict': model.state_dict()}, os.path.join(args.output_dir, 'checkpoint.pt'))
            torch.save({'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()}, os.path.join(args.output_dir, 'optim_checkpoint.pt'))

# --------------------
# Training helper functions
# --------------------

def build_lr(step, steps_per_epoch, mode='exponential', decay_epochs=2.4, warmup_epochs=5, decay_factor=0.97):
    # provides the multiplicative factor to LambdaLR scheduler, which then returns initial_lr * factor to the optimizer
    if warmup_epochs:
        warmup_steps = int(warmup_epochs * steps_per_epoch)
        if step < warmup_steps:
            return step / warmup_steps

    if mode=='constant':
        return 1
    elif mode=='exponential':
        # decay
        decay_steps = decay_epochs * steps_per_epoch
        return decay_factor ** (step // decay_steps)  # staircase decay (cf tf.train.exponential_decay)
    else:
        raise 'Invalid learning rate scheduling mode.'


if __name__ == '__main__':
    args = parser.parse_args()

    # set up output folders
    if not args.output_dir and not args.restore: args.output_dir = os.path.join('results', time.strftime('%Y-%m-%d_%H-%M-%S', time.gmtime()))
    writer = SummaryWriter(logdir=args.output_dir)  # creates output_dir

    args.device = torch.device('cuda:{}'.format(args.cuda) if args.cuda is not None and torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)
    if args.device.type == 'cuda':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # setup model
    model = construct_model(args.model, n_classes=10).to(args.device)
    print(model)
    print('Loaded {} (number of parameters: {:,})'.format(model.__class__.__name__, sum(p.numel() for p in model.parameters())))

    # save config
    if not os.path.exists(os.path.join(args.output_dir, 'config.txt')):
        with open(os.path.join(args.output_dir, 'config.txt'), 'w') as f:
            print(pprint.pformat(args.__dict__), file=f)
            print('Model architecture:\n' + str(model), file=f)

    # dataloader
    dataset = partial(CIFAR10, root=args.data_dir, transform=T.ToTensor())
    if args.mini_data:
        dataset = dataset(train=True)
        dataset.data = dataset.data[:args.batch_size]
        dataset.targets = dataset.targets[:args.batch_size]
        train_dataloader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=4, pin_memory=(args.device.type=='cuda'))
        valid_dataloader = train_dataloader
    else:
        train_dataloader = DataLoader(dataset(train=True), args.batch_size, shuffle=True, num_workers=4, pin_memory=(args.device.type=='cuda'))
        valid_dataloader = DataLoader(dataset(train=False), args.batch_size, shuffle=False, num_workers=4, pin_memory=(args.device.type=='cuda'))

    # loss
    loss_fn = nn.CrossEntropyLoss().to(args.device)

    if args.restore:
        print('Restoring model weights from {}'.format(args.restore))
        model_checkpoint = torch.load(args.restore, map_location=args.device)
        model.load_state_dict(model_checkpoint['state_dict'])
        args.step = model_checkpoint['global_step']

    if args.train:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9, eps=0.001)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, partial(build_lr, steps_per_epoch=len(train_dataloader)))
        if args.restore:
            optim_checkpoint_path = os.path.join(os.path.dirname(args.restore), 'optim_' + os.path.basename(args.restore))
            optim_checkpoint = torch.load(optim_checkpoint_path, map_location=args.device)
            optimizer.load_state_dict(optim_checkpoint['optimizer'])
            scheduler.load_state_dict(optim_checkpoint['scheduler'])

        train_and_evaluate(model, train_dataloader, valid_dataloader, loss_fn, optimizer, scheduler, writer, args)

    if args.evaluate:
        print('Evaluating...', end='\r')
        loss, top1, top5 = evaluate(model, valid_dataloader, loss_fn, args)
        print('Evaluate @ step {}: loss {:.4f}; acc@1 {:.4f}; acc@5 {:.4f}'.format(args.step, loss, top1, top5))

    writer.close()
