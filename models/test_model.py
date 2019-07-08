import os
import time
import json
import pprint
import argparse
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10, CIFAR100
import torchvision.transforms as T
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from efficientnet import construct_model
from attn_aug_conv import ResNet, WideResNet, Bottleneck, BasicBlock, DenseNet
resnet_layers = {50: [3,4,6,3], 101: [3,4,23,3], 152: [3,8,36,3]}


parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest='model', help='Select model architecture.', required=True)

# models
#   1. efficientnet
parser_a = subparsers.add_parser('efficientnet', description='EfficientNet B0-7 (https://arxiv.org/pdf/1905.11946.pdf)')
parser_a.add_argument('architecture', default='b0', choices=['b0','b1','b2','b3','b4','b5','b6','b7'], help='Efficientnet architecture.')
#   2. resnet
parser_b = subparsers.add_parser('resnet', description='ResNet50 torchvision implementation.')
parser_b.add_argument('architecture', type=int, default=50, choices=[50, 101, 152], help='Resnet architecture.')
#   4. wideresnet
parser_c = subparsers.add_parser('wideresnet', description='WideResNet (https://arxiv.org/pdf/1605.07146.pdf).')
parser_c.add_argument('architecture', type=int, default=[28, 10], nargs=2, help='WideResnet depth and width.')
#   6. densenet
parser_d = subparsers.add_parser('densenet', description='Attention augmented DenseNet-BC')
parser_d.add_argument('architecture', type=int, default=[12, 100], nargs=2, help='Densenet growth (k) and depth (L) parameters.')

# attention parameters cf Attention Augmented Convolutions https://arxiv.org/pdf/1904.09925.pdf
parser.add_argument('--attn', action='store_true', default=False, help='Attention augmented architecture.')
parser.add_argument('--attn_k', type=float, default=0.2, help='Ration of dk/out_channels (keys dimension / out_channels).')
parser.add_argument('--attn_v', type=float, default=0.1, help='Ration of dv/out_channels (values dimension / out_channels).')
parser.add_argument('--attn_nh', type=int, default=8, help='Number of self-attention heads.')
parser.add_argument('--attn_relative', type=eval, default=True, help='Whether to use relative positional encoding.')
parser.add_argument('--input_dims', default=(32,32), type=int, nargs='+', help='Dimensions of the input data used for relative positional encodings.')

# actions
parser.add_argument('--load_config', type=str, help='Path to config.json file to load args from.')
parser.add_argument('--train', action='store_true', help='Train model.')
parser.add_argument('--evaluate', action='store_true', help='Evaluate a single model.')
parser.add_argument('--vis_attn', action='store_true', help='Visualize the attention maps of a model.')
parser.add_argument('--seed', type=int, default=0, help='Random seed to use.')
parser.add_argument('--cuda', type=int, help='Which cuda device to use.')
parser.add_argument('--mini_data', action='store_true', help='Truncate dataset to a single batch.')

# paths
parser.add_argument('--dataset', default='cifar100', choices=['cifar10', 'cifar100'], help='Dataset to use.')
parser.add_argument('--data_dir', default='~/data/cifar100/', help='Location of datasets.')
parser.add_argument('--output_dir', help='Path to experiment output, config, checkpoints, etc.')
parser.add_argument('--restore', type=str, help='Path to a single model checkpoint to restore.')

# training params
parser.add_argument('--batch_size', type=int, default=256, help='Dataloaders batch size.')
parser.add_argument('--n_epochs', type=int, default=1, help='Number of epochs to train.')
parser.add_argument('--step', type=int, default=0, help='Current step of training (number of minibatches processed).')
parser.add_argument('--log_interval', type=int, default=1, help='Interval of num batches to show loss statistics.')
parser.add_argument('--eval_interval', type=int, default=10, help='Interval of epochs to evaluate and save model.')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay regularization.')
parser.add_argument('--lr', type=float, default=0.016, help='Base learning rate when the batch size is 256.')
parser.add_argument('--lr_warmup_epochs', type=int, default=5, help='Warmup from 0 to base learning rate for this many epochs.')
parser.add_argument('--lr_cos_max_epochs', type=int, default=25, help='Maximum number of interations of cosine annealing if cosine scheduler.')
parser.add_argument('--lr_decay_factor', type=float, default=0.97, help='Learning rate decay factor if exponential scheduler.')
parser.add_argument('--lr_decay_epochs', type=float, default=2.4, help='How often to decay the learning rate if exponential scheduler.')


# --------------------
# Data IO
# --------------------

def save_json(data, filename, args):
    with open(os.path.join(args.output_dir, filename + '.json'), 'w') as f:
        json.dump(data, f, indent=4)

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

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
# Learning rate schedulers
# --------------------

class ExponentialLR(torch.optim.lr_scheduler.ExponentialLR):
    def __init__(self, optimizer, gamma, decay_steps):
        self.decay_steps = decay_steps
        super().__init__(optimizer, gamma)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        return [group['lr'] * self.gamma ** (self.last_epoch // self.decay_steps)  # staircase decay (cf tf.train.exponential_decay)
                for group in self.optimizer.param_groups]

def build_scheduler(scheduler_class, warmup_steps, *args, **kwargs):
    """ scheduler wrapper to enable warmup; args and kwargs are passed to initialize scheduler class """
    class Scheduler(scheduler_class):
        def __init__(self, warmup_steps, *args, **kwargs):
            self.warmup_steps = warmup_steps
            super().__init__(*args, **kwargs)
        def get_lr(self):
            if self.last_epoch < self.warmup_steps:
                return [base_lr * self.last_epoch / self.warmup_steps for base_lr in self.base_lrs]
            return super().get_lr()
    return Scheduler(warmup_steps, *args, **kwargs)


# --------------------
# Visualize attention
# --------------------

def vis_attn(x, layers, args, batch_element=0):
    for j, l in enumerate(layers):
        h, w = x.shape[2:]
        nh = layers[0].nh

        # select which pixels to visualize -- e.g. select virtices of a center square of side 1/3 of the image dims
        pix_to_vis = lambda h, w: [(h//3, w//3), (h//3, int(2*w/3)), (int(2*h/3), w//3), (int(2*h/3), int(2*w/3))]

        # visualize attention maps (rows for each head; columns for each pixel)
        fig, axs = plt.subplots(nh+1, 4, figsize=(3,3/4*(1+nh)), frameon=False)
        # display target image; highlight pixel
        for ax, (ph, pw) in zip(axs[0],pix_to_vis(*x.shape[2:])):
            image = x.clone()
            image[:,:,ph,pw] = torch.tensor([1., 215/255, 0])
            ax.imshow(image[batch_element].permute(1,2,0).numpy())
            ax.axis('off')
        # display attention maps
        # grab attention weights tensor for the batch element
        attn = l.weights.data[batch_element]
        # reshape attn tensor and select the pixels to visualize
        h = w = int(np.sqrt(attn.shape[-1]))
        attn = attn.reshape(nh, h, w, h, w)
        for i, (ph, pw) in enumerate(pix_to_vis(h,w)):
            for h in range(nh):
                axs[h+1, i].imshow(attn[h, ph, pw, :, :].numpy())
                axs[h+1, i].axis('off')

        filename = 'vis_attn_image_{}_layer_{}.png'.format(batch_element, j)
        fig.subplots_adjust(0,0,1,1,0.05,0.05)
        plt.savefig(os.path.join(args.output_dir, filename))
        plt.close()

# --------------------
# Run
# --------------------

if __name__ == '__main__':
    args = parser.parse_args()

    # overwrite args from config
    if args.load_config:
        config = load_json(args.load_config)
        del config['output_dir']
        args.__dict__.update(config)
        args.output_dir = os.path.dirname(args.load_config)

    # set up output folder
    if not args.output_dir:
        args.output_dir = os.path.dirname(args.restore) if args.restore else \
                            os.path.join('results', args.model, time.strftime('%Y-%m-%d_%H-%M-%S', time.gmtime()))
    writer = SummaryWriter(logdir=args.output_dir)  # creates output_dir

    # save config
    if not os.path.exists(os.path.join(args.output_dir, 'config.json')): save_json(args.__dict__, 'config', args)
    writer.add_text('config', str(args.__dict__))

    args.device = torch.device('cuda:{}'.format(args.cuda) if args.cuda is not None and torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)
    if args.device.type == 'cuda':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # setup dataset and dataloader
    dataset = partial(CIFAR10 if args.dataset.lower()=='cifar10' else CIFAR100, root=args.data_dir)
    valid_transforms = T.Compose([T.ToTensor(), T.Normalize([125.3/255,123.0/255,113.9/255],[63.0/255,62.1/255,66.7/255])])
    train_transforms = T.Compose([T.Pad(4, padding_mode='reflect'), T.RandomHorizontalFlip(), T.RandomCrop(32), valid_transforms])
    if args.mini_data:
        dataset = dataset(train=True, transform=valid_transforms)
        dataset.data = dataset.data[:args.batch_size]
        dataset.targets = dataset.targets[:args.batch_size]
        train_dataloader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=4, pin_memory=(args.device.type=='cuda'))
        valid_dataloader = train_dataloader
    else:
        train_dataloader = DataLoader(dataset(train=True, transform=train_transforms), args.batch_size, shuffle=True, num_workers=4, pin_memory=(args.device.type=='cuda'))
        valid_dataloader = DataLoader(dataset(train=False, transform=valid_transforms), args.batch_size, shuffle=False, num_workers=4, pin_memory=(args.device.type=='cuda'))

    # setup model
    n_classes = 10 if args.dataset.lower()=='cifar10' else 100
    n_batches = len(train_dataloader)
    if args.model=='efficientnet':
        model = construct_model(args.model+'-'+args.architecture, n_classes=n_classes).to(args.device)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, momentum=0.9, eps=0.001)
        scheduler = build_scheduler(ExponentialLR, args.lr_warmup_epochs*n_batches, optimizer,
                        gamma=args.lr_decay_factor, decay_steps=args.lr_decay_epochs*n_batches)  # scheduler-specific kwargs
    elif args.model=='resnet':
        model = ResNet(Bottleneck, resnet_layers[args.architecture], num_classes=n_classes,
                       attn_params=None if not args.attn else \
                                   {'k': args.attn_k, 'v': args.attn_v, 'nh': args.attn_nh,
                                    'relative': args.attn_relative, 'input_dims': args.input_dims}).to(args.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9, nesterov=True)
        scheduler = build_scheduler(torch.optim.lr_scheduler.CosineAnnealingLR, args.lr_warmup_epochs*n_batches, optimizer,
                        T_max=args.lr_cos_max_epochs*n_batches)  # scheduler-specific kwargs
    elif args.model=='wideresnet':
        model = WideResNet(BasicBlock, *args.architecture, num_classes=n_classes,
                           attn_params=None if not args.attn else \
                                       {'k': args.attn_k, 'v': args.attn_v, 'nh': args.attn_nh,
                                        'relative': args.attn_relative, 'input_dims': args.input_dims}).to(args.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9, nesterov=True)
        scheduler = build_scheduler(torch.optim.lr_scheduler.CosineAnnealingLR, args.lr_warmup_epochs*n_batches, optimizer,
                        T_max=args.lr_cos_max_epochs*n_batches)  # scheduler-specific kwargs
    elif args.model=='densenet':
        k, L = args.architecture
        model = DenseNet(k, ((L-4)//6, (L-4)//6, (L-4)//6), 2*k, num_classes=n_classes,
                         attn_params=None if not args.attn else \
                                     {'k': args.attn_k, 'v': args.attn_v, 'nh': args.attn_nh,
                                      'relative': args.attn_relative, 'input_dims': args.input_dims}).to(args.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9, nesterov=True)
        scheduler = build_scheduler(torch.optim.lr_scheduler.MultiStepLR, args.lr_warmup_epochs*n_batches, optimizer,
                        milestones=[100*n_batches, 150*n_batches])  # scheduler-specific kwargs
    else:
        raise RuntimeError('Model not supported.')

#    print(model)
    pprint.pprint(args.__dict__)
    print('Loaded {} (number of parameters: {:,})'.format(args.model+'-'+str(args.architecture), sum(p.numel() for p in model.parameters())))

    if args.restore:
        print('Restoring model weights from {}'.format(args.restore))
        model_checkpoint = torch.load(args.restore, map_location=args.device)
        model.load_state_dict(model_checkpoint['state_dict'])
        args.step = model_checkpoint['global_step']
        print('Restoring optimizer and scheduler.')
        optim_checkpoint_path = os.path.join(os.path.dirname(args.restore), 'optim_' + os.path.basename(args.restore))
        optim_checkpoint = torch.load(optim_checkpoint_path, map_location=args.device)
        optimizer.load_state_dict(optim_checkpoint['optimizer'])
        scheduler.load_state_dict(optim_checkpoint['scheduler'])

    loss_fn = nn.CrossEntropyLoss().to(args.device)

    if args.train:
        train_and_evaluate(model, train_dataloader, valid_dataloader, loss_fn, optimizer, scheduler, writer, args)

    if args.evaluate:
        print('Evaluating...', end='\r')
        loss, top1, top5 = evaluate(model, valid_dataloader, loss_fn, args)
        print('Evaluate @ step {}: loss {:.4f}; acc@1 {:.4f}; acc@5 {:.4f}'.format(args.step, loss, top1, top5))

    if args.vis_attn:
        assert args.attn==True, 'Enable --attn flag to visualize attention.'
        # process a batch of data through the model (stores attention values)
        x = next(iter(valid_dataloader))[0][:8].to(args.device)
        model(x)

        # list attention layers to visualize
        if args.model=='wideresnet':
            layers = [model.layer2[i].conv1 for i in range(len(model.layer2))] + \
                     [model.layer3[i].conv1 for i in range(len(model.layer3))]
        elif args.model=='densenet':
            layers = [model.features.transition1.conv, model.features.transition2.conv]
        else:
            raise RuntimeError('Model not supported.')

        # invert the data normalization before visualizing raw images
        transforms = T.Compose([T.Normalize([0,0,0],[255/63.0,255/62.1,255/66.7]),
                                T.Normalize([-125.3/255,-123.0/255,-113.9/255],[1, 1, 1])])
        images = torch.stack([transforms(x[i]) for i in range(len(x))], 0)

        # visualize stored attention weights for each image
        for i in range(len(x)): vis_attn(images, layers, args, i)

    writer.close()
