import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.models import densenet121, resnet152

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from tqdm import tqdm
from tensorboardX import SummaryWriter

import os
import pprint
import argparse
import time
import json
from collections import OrderedDict

from dataset import ChexpertSmall


parser = argparse.ArgumentParser()
# action
parser.add_argument('--train', action='store_true', help='Train model.')
parser.add_argument('--evaluate_single_model', action='store_true', help='Evaluate a single model.')
parser.add_argument('--evaluate_ensemble', action='store_true', help='Evaluate an ensemble (given a checkpoints tracker of saved model checkpoints).')
parser.add_argument('--visualize', action='store_true', help='Visualize Grad-CAM.')
parser.add_argument('--plot_roc', action='store_true', help='Filename for metrics json file to plot ROC.')
parser.add_argument('--seed', type=int, default=0, help='Random seed to use.')
parser.add_argument('--cuda', type=int, help='Which cuda device to use.')
# paths
parser.add_argument('--data_path', default='', help='Location of datasets.')
parser.add_argument('--output_dir', help='Path to experiment output, config, checkpoints, etc.')
parser.add_argument('--vis_dir', default='vis', help='Path relative to output_dir, where visualization samples are saved.')
parser.add_argument('--plots_dir', default='plots', help='Path relative to output_dir, where roc pr plots are saved.')
parser.add_argument('--best_checkpoints_dir', default='best_checkpoints', help='Path relative to output_dir, where best checkpoints are saved.')
parser.add_argument('--restore', type=str, help='Path relative to output_dir pointing to a single model checkpoint to restore or folder of checkpoints to ensemble.')
parser.add_argument('--metrics_file', type=str, help='Path to metrics json file to plot results.')
# model
parser.add_argument('--model', default='densenet121', choices=['densenet121', 'resnet152'], help='What model architecture to use.')
# data params
parser.add_argument('--mini_data', type=int, help='Truncate dataset to first entry only.')
parser.add_argument('--resize', type=int, help='Size of minimum edge to which to resize images.')
# training params
parser.add_argument('--pretrained', action='store_true', help='Use ImageNet pretrained model and normalize data mean and std.')
parser.add_argument('--batch_size', type=int, default=16, help='Dataloaders batch size.')
parser.add_argument('--n_epochs', type=int, default=1, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
parser.add_argument('--step', type=int, default=0, help='Current step of training (number of minibatches processed).')
parser.add_argument('--log_interval', type=int, default=50, help='Interval of num batches to show loss statistics.')
parser.add_argument('--eval_interval', type=int, default=300, help='Interval of num epochs to evaluate, checkpoint, and save samples.')


# --------------------
# Data IO
# --------------------

class CustomBatch:
    """ Custom batch class so can pin memory to gpu; Note - target is None if in test mode """
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.img = torch.stack(transposed_data[0], 0)
        self.target = torch.stack(transposed_data[1], 0) if not any(x is None for x in transposed_data[1]) else None
        self.patient_id = transposed_data[2]

    def pin_memory(self):
        self.img = self.img.pin_memory()
        self.target = self.target.pin_memory() if self.target is not None else self.target
        return self

def collate_wrapper(batch):
    return CustomBatch(batch)

def fetch_dataloader(args, mode):

    transforms = T.Compose([
        T.Resize(args.resize) if args.resize else T.Lambda(lambda x: x),
        T.CenterCrop(320),
        T.ToTensor(),
        T.Lambda(lambda x: x.expand(3,-1,-1)),  # expand to 3 channels
        T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]) if args.pretrained else T.Lambda(lambda x: x)])

    dataset = ChexpertSmall(args.data_path, mode, transforms, args.mini_data)

    return DataLoader(dataset, args.batch_size, shuffle=True if mode=='train' else False, collate_fn=collate_wrapper,
                      pin_memory=True if args.device.type is 'cuda' else False, num_workers=4)

def save_json(data, filename, args):
    with open(os.path.join(args.output_dir, filename + '.json'), 'w') as f:
        json.dump(data, f, indent=4)

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def save_checkpoint(checkpoint, optim_checkpoint, args, max_records=10):
    """ save model and optimizer checkpoint along with csv tracker of last `max_records` number of checkpoints
    as sorted by avg auc """
    # 1. save latest
    torch.save(checkpoint, os.path.join(args.output_dir, 'checkpoint_latest.pt'))
    torch.save(optim_checkpoint, os.path.join(args.output_dir, 'optim_checkpoint_latest.pt'))

    # 2. save the last `max_records` number of checkpoints as sorted by avg auc
    tracker_path = os.path.join(args.output_dir, 'checkpoints_tracker.csv')
    tracker_header = ' '.join(['CheckpointId', 'Step', 'Loss', 'AvgAUC'])

    # 2a. load checkpoint stats from file
    old_data = None             # init and overwrite from records
    file_id = 0                 # init and overwrite from records
    lowest_auc = float('-inf')  # init and overwrite from records
    if os.path.exists(tracker_path):
        old_data = np.atleast_2d(np.loadtxt(tracker_path, skiprows=1))
        file_id = len(old_data)
        if len(old_data) == max_records: # remove the lowest-roc record and add new checkpoint record under its file-id
            lowest_auc_idx = old_data[:,3].argmin()
            lowest_auc = old_data[lowest_auc_idx, 3]
            file_id = int(old_data[lowest_auc_idx, 0])
            old_data = np.delete(old_data, lowest_auc_idx, 0)

    # 2b. update tracking data and sort by descending avg auc
    data = np.atleast_2d([file_id, args.step, checkpoint['eval_loss'], checkpoint['avg_auc']])
    if old_data is not None: data = np.vstack([old_data, data])
    data = data[data.argsort(0)[:,3][::-1]]  # sort descending by AvgAUC column

    # 2c. save tracker and checkpoint if better than what is already saved
    if checkpoint['avg_auc'] > lowest_auc:
        np.savetxt(tracker_path, data, delimiter=' ', header=tracker_header)
        torch.save(checkpoint, os.path.join(args.output_dir, args.best_checkpoints_dir, 'checkpoint_{}.pt'.format(file_id)))


# --------------------
# Evaluation metrics
# --------------------

def compute_metrics(outputs, targets, losses):
    outputs, targets, losses = outputs.cpu(), targets.cpu(), losses.cpu()

    n_classes = outputs.shape[1]
    fpr, tpr, aucs, precision, recall = {}, {}, {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(targets[:,i], outputs[:,i])
        aucs[i] = auc(fpr[i], tpr[i])
        precision[i], recall[i], _ = precision_recall_curve(targets[:,i], outputs[:,i])
        fpr[i], tpr[i], precision[i], recall[i] = fpr[i].tolist(), tpr[i].tolist(), precision[i].tolist(), recall[i].tolist()

    metrics = {'fpr': fpr,
               'tpr': tpr,
               'aucs': aucs,
               'precision': precision,
               'recall': recall,
               'loss': dict(enumerate(losses.mean(0).tolist()))}

    return metrics

# --------------------
# Train and evaluate
# --------------------

def train_epoch(model, train_dataloader, valid_dataloader, loss_fn, optimizer, writer, epoch, args):
    with tqdm(total=len(train_dataloader), desc='Step at start {}; Training epoch {}/{}'.format(args.step, epoch+1, args.n_epochs)) as pbar:
        for batch in train_dataloader:
            model.train()
            args.step += 1

            x = batch.img.to(args.device)
            y = batch.target.to(args.device)

            outputs = model(x)
            loss = loss_fn(outputs, y).sum(1).mean(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss = '{:.4f}'.format(loss.item()))
            pbar.update()

            # record
            if args.step % args.log_interval == 0:
                writer.add_scalar('train_loss', loss.item(), args.step)

            # evaluate and save on eval_interval
            if args.step % args.eval_interval == 0:
                eval_metrics = evaluate_single_model(model, valid_dataloader, loss_fn, args)

                writer.add_scalar('eval_loss', np.mean(list(eval_metrics['loss'].values())), args.step)
                for k, v in eval_metrics['aucs'].items():
                    writer.add_scalar('eval_auc_class_{}'.format(k), v, args.step)

                # save model
                checkpoint = {'global_step': args.step,
                              'eval_loss': np.mean(list(eval_metrics['loss'].values())),
                              'avg_auc': np.nanmean(list(eval_metrics['aucs'].values())),
                              'state_dict': model.state_dict()}
                optim_checkpoint = optimizer.state_dict()
                save_checkpoint(checkpoint, optim_checkpoint, args)

                # visualize grad-cam and plot roc
                visualize(model, valid_dataloader, args)
                plot_roc(eval_metrics, args)

@torch.no_grad()
def evaluate(model, dataloader, loss_fn, args):
    model.eval()

    targets, outputs, losses = [], [], []
    for batch in dataloader:
        x = batch.img.to(args.device)
        y = batch.target.to(args.device)

        out = model(x)
        loss = loss_fn(out, y)

        outputs += [out]
        targets += [y]
        losses  += [loss]

    return torch.cat(outputs, 0), torch.cat(targets, 0), torch.cat(losses, 0)

def evaluate_single_model(model, dataloader, loss_fn, args):
    outputs, targets, losses = evaluate(model, dataloader, loss_fn, args)
    return compute_metrics(outputs, targets, losses)

def evaluate_ensemble(model, dataloader, loss_fn, args):
    checkpoints = [c for c in os.listdir(os.path.join(args.output_dir, args.restore)) \
                        if c.startswith('checkpoint') and c.endswith('.pt')]
    print('Running ensemble prediction using {} checkpoints.'.format(len(checkpoints)))
    outputs, losses = [], []
    for checkpoint in checkpoints:
        # load weights
        model_checkpoint = torch.load(os.path.join(args.output_dir, args.restore, checkpoint), map_location=args.device)
        model.load_state_dict(model_checkpoint['state_dict'])
        # evaluate
        outputs_, targets, losses_ = evaluate(model, dataloader, loss_fn, args)
        outputs += [outputs_]
        losses  += [losses_]

    # take mean over checkpoints
    outputs  = torch.stack(outputs, dim=2).mean(2)
    losses = torch.stack(losses, dim=2).mean(2)

    return compute_metrics(outputs, targets, losses)

def train_and_evaluate(model, train_dataloader, valid_dataloader, loss_fn, optimizer, writer, args):
    for epoch in range(args.n_epochs):
        # train
        train_epoch(model, train_dataloader, valid_dataloader, loss_fn, optimizer, writer, epoch, args)

        # evaluate
        print('Evaluating...', end='\r')
        eval_metrics = evaluate_single_model(model, valid_dataloader, loss_fn, args)
        print('Evaluate metrics @ step {}:'.format(args.step))
        print('AUC:\n', pprint.pformat(eval_metrics['aucs']))
        print('Loss:\n', pprint.pformat(eval_metrics['loss']))
        writer.add_scalar('eval_loss', np.mean(list(eval_metrics['loss'].values())), args.step)
        for k, v in eval_metrics['aucs'].items():
            writer.add_scalar('eval_auc_class_{}'.format(k), v, args.step)

        # save eval metrics
        save_json({'step': args.step, 'eval_metrics': eval_metrics}, 'eval_results_step_{}'.format(args.step), args)

# --------------------
# Visualization
# --------------------

def grad_cam(model, x, hooks, cls_idx=None):
    """ cf CheXpert: Test Results / Visualization; visualize final conv layer, using grads of final linear layer as weights,
    and performing a weighted sum of the final feature maps using those weights.
    cf Grad-CAM https://arxiv.org/pdf/1610.02391.pdf """

    model.eval()
    model.zero_grad()

    # register backward hooks
    conv_features, linear_grad = [], []
    #model.features.denseblock4.denselayer16.conv2.register_forward_hook(lambda module, in_tensor, out_tensor: conv_features.append(out_tensor))
    forward_handle = hooks['forward'].register_forward_hook(lambda module, in_tensor, out_tensor: conv_features.append(out_tensor))
    backward_handle = hooks['backward'].register_backward_hook(lambda module, grad_input, grad_output: linear_grad.append(grad_input))

    # run model forward and create a one hot output for the given cls_idx or max class
    outputs = model(x)

    if not cls_idx:
        cls_idx = outputs.argmax(1)
    one_hot = torch.zeros_like(outputs)
    one_hot[torch.arange(outputs.shape[0]), cls_idx] = 1
    one_hot.requires_grad_(True)

    # run model backward
    one_hot.mul(outputs).sum().backward()

    # compute weights; cf. Grad-CAM eq 1 -- gradients flowing back are global-avg-pooled to obtain the neuron importance weights
    weights = linear_grad[0][2].mean(1).view(1, -1, 1, 1)
    # compute weighted combination of forward activation maps; cf Grad-CAM eq 2; linear combination over channels
    cam = F.relu(torch.sum(weights * conv_features[0], dim=1, keepdim=True))

    # normalize each image in the minibatch to [0,1] and upscale to input image size
    cam = cam.clone()  # avoid modifying tensor in-place
    def norm_ip(t, min, max):
        t.clamp_(min=min, max=max)
        t.add_(-min).div_(max - min + 1e-5)

    for t in cam:  # loop over mini-batch dim
        norm_ip(t, float(t.min()), float(t.max()))

    cam = F.interpolate(cam, x.shape[2:], mode='bilinear', align_corners=True)

    forward_handle.remove()
    backward_handle.remove()

    return cam

def visualize(model, dataloader, args, n_samples=20):
    """ visualize grad-cam on examples """
    # load examples for dataloader
    batch = next(iter(dataloader))
    xs, labels, patient_ids = batch.img, batch.target, batch.patient_id
    xs = xs[0:n_samples].to(args.device)
    labels = labels[0:n_samples]
    attr_names = dataloader.dataset.attr_names

    # compute logits and grad-cam
    scores = model(xs)
    masks = grad_cam(model, xs, args.hooks)

    # apply mask grad-cam on each image in the minibatch and save
    # 1. renormalize if using a pretrained model
    if args.pretrained: xs.mul_(torch.tensor([0.229, 0.224, 0.225], device=args.device).view(1,3,1,1))\
                          .add_(torch.tensor([0.485, 0.456, 0.406], device=args.device).view(1,3,1,1))
    # 2. move everything to cpu
    xs = xs.cpu().permute(0,2,3,1).data.numpy()
    masks = masks.cpu().permute(0,2,3,1).data.numpy()
    labels = labels.cpu().data.numpy()
    probs = scores.sigmoid().cpu().data.numpy()
    # 3. apply grad-cam mask and save
    for i, (x, mask, label, patient_id, prob) in enumerate(zip(xs, masks, labels, patient_ids, probs)):
        # sort data by prob high to low
        sort_idxs = prob.argsort()[::-1]
        label = label[sort_idxs]
        prob = prob[sort_idxs]
        names = [attr_names[i] for i in sort_idxs]
        # set up figure
        fig, axs = plt.subplots(1, 4, figsize=(4 * xs.shape[1]/100, xs.shape[2]/100), dpi=100, frameon=False)
        fig.suptitle(patient_id)
        # 1. left -- show table of ground truth and predictions, sorted by pred prob high to low
        data = np.stack([label, prob.round(3)]).T
        axs[1].table(cellText=data, rowLabels=names, colLabels=['GT', 'Pred. prob'],
                     rowColours=plt.cm.Greens(0.5*label),
                     cellColours=plt.cm.Greens(0.5*data), loc='center')
        axs[1].axis('tight')
        # 2. middle -- show original image
        axs[2].set_title('Original image', fontsize=10)
        axs[2].imshow(x.squeeze(), cmap='gray')
        # 3. right -- show heatmap over original image with predictions
        axs[3].set_title('Top class activation \n{}: {:.4f}'.format(names[0], prob[0]), fontsize=10)
        axs[3].imshow(x.squeeze(), cmap='gray')
        axs[3].imshow(mask.squeeze(), cmap='jet', alpha=0.5)

        for ax in axs: ax.axis('off')

        filename = 'vis_{}_view_{}_trainstep_{}.png'.format('-'.join(patient_id.split('/')), i, args.step)
        plt.savefig(os.path.join(args.output_dir, args.vis_dir, filename), dpi=100)
        plt.close()

def plot_roc(metrics, args, labels=ChexpertSmall.attr_names):
    fig, axs = plt.subplots(2, len(labels), figsize=(24,12))

    for i, (fpr, tpr, aucs, precision, recall, label) in enumerate(zip(metrics['fpr'].values(), metrics['tpr'].values(), 
                                                                       metrics['aucs'].values(), metrics['precision'].values(),
                                                                       metrics['recall'].values(), labels)):
        # top row -- ROC
        axs[0,i].plot(fpr, tpr, label='AUC = %0.2f' % aucs)
        axs[0,i].plot([0, 1], [0, 1], 'k--')  # diagonal margin
        axs[0,i].set_xlabel('False Positive Rate')
        # bottom row - Precision-Recall
        axs[1,i].step(recall, precision, where='post')
        axs[1,i].set_xlabel('Recall')
        # format
        axs[0,i].set_title(label)
        axs[0,i].legend(loc="lower right")

    axs[0,0].set_ylabel('True Positive Rate')
    axs[1,0].set_ylabel('Precision')

    for ax in axs.flatten():
        ax.set_xlim([0.0, 1.05])
        ax.set_ylim([0.0, 1.05])
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, args.plots_dir, 'roc_pr_step_{}.png'.format(args.step)), pad_inches=0.)
    plt.close()

# --------------------
# Main
# --------------------

if __name__ == '__main__':
    args = parser.parse_args()

    # set up output folders
    if not args.output_dir: args.output_dir = os.path.join('results', time.strftime('%Y-%m-%d_%H-%M-%S', time.gmtime()))
    # make new folders if they don't exist
    writer = SummaryWriter(logdir=args.output_dir)  # creates output_dir
    if not os.path.exists(os.path.join(args.output_dir, args.vis_dir)): os.makedirs(os.path.join(args.output_dir, args.vis_dir))
    if not os.path.exists(os.path.join(args.output_dir, args.plots_dir)): os.makedirs(os.path.join(args.output_dir, args.plots_dir))
    if not os.path.exists(os.path.join(args.output_dir, args.best_checkpoints_dir)): os.makedirs(os.path.join(args.output_dir, args.best_checkpoints_dir))

    # save config
    if not os.path.exists(os.path.join(args.output_dir, 'config.json')): save_json(args.__dict__, 'config', args)

    args.device = torch.device('cuda:{}'.format(args.cuda) if args.cuda is not None and torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device.type == 'cuda': torch.cuda.manual_seed(args.seed)

    # load model
    n_classes = len(ChexpertSmall.attr_names)
    model = locals()[args.model](pretrained=args.pretrained)
    # 1. replace output layer with chexpert number of classes
    # 2. init output layer with default torchvision init
    # 3. store locations of forward and backward hooks for grad-cam
    if 'densenet' in args.model:
        model.classifier = nn.Linear(model.classifier.in_features, out_features=n_classes)
        nn.init.constant_(model.classifier.bias, 0)
        args.hooks = {'forward': model.features.norm5, 'backward': model.classifier}
    if 'resnet' in args.model:
        model.fc = nn.Linear(model.fc.in_features, out_features=n_classes)
        args.hooks = {'forward': model.avgpool, 'backward': model.fc}
    model = model.to(args.device)

    if args.restore:
        if os.path.isfile(os.path.join(args.output_dir, args.restore)):  # restore from single file, else ensemble is handled by evaluate_ensemble
            print('Restoring model weights from {}'.format(args.restore))
            model_checkpoint = torch.load(os.path.join(args.output_dir, args.restore), map_location=args.device)
            model.load_state_dict(model_checkpoint['state_dict'])
            args.step = model_checkpoint['global_step']
        # load pretrained from config -- in case args.pretrained flag is forgotten e.g. in post-training evaluation
        # (images still need to be normalized if training started on an imagenet pretrained model)
        args.pretrained = load_json(os.path.join(args.output_dir, 'config.json'))['pretrained']

    # setup loss function for train and eval
    loss_fn = nn.BCEWithLogitsLoss(reduction='none').to(args.device)

    # load data (after pretrained flag for data normalization to ImageNet is loaded from config)
    train_dataloader = fetch_dataloader(args, mode='train')
    valid_dataloader = fetch_dataloader(args, mode='valid')

    print('Loaded {} (number of parameters: {:,}; weights trained to step {})'.format(
        model._get_name(), sum(p.numel() for p in model.parameters()), args.step))
    print('Train data length: ', len(train_dataloader) * args.batch_size)
    print('Valid data length: ', len(valid_dataloader) * args.batch_size)

    if args.train:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        if os.path.isfile(args.restore):
            optimizer.load_state_dict(torch.load(os.path.join(args.output_dir, 'optim_' + args.restore), map_location=args.device))
        train_and_evaluate(model, train_dataloader, valid_dataloader, loss_fn, optimizer, writer, args)

    if args.evaluate_single_model:
        eval_metrics = evaluate_single_model(model, valid_dataloader, loss_fn, args)
        print('Evaluate metrics -- \n\t restore: {} \n\t step: {}:'.format(args.restore, args.step))
        print('AUC:\n', pprint.pformat(eval_metrics['aucs']))
        print('Loss:\n', pprint.pformat(eval_metrics['loss']))
        save_json({'step': args.step, 'eval_metrics': eval_metrics}, 'eval_results_step_{}'.format(args.step), args)

    if args.evaluate_ensemble:
        eval_metrics = evaluate_ensemble(model, valid_dataloader, loss_fn, args)
        print('Evaluate ensemble metrics -- \n\t checkpoints path {}:'.format(os.path.join(args.output_dir, args.restore)))
        print('AUC:\n', pprint.pformat(eval_metrics['aucs']))
        print('Loss:\n', pprint.pformat(eval_metrics['loss']))
        save_json(eval_metrics, 'eval_ensemble_results', args)

    if args.visualize:
        visualize(model, valid_dataloader, args)

    if args.plot_roc:
        if 'eval_metrics' not in locals():
            eval_metrics = load_json(args.metrics_file)['eval_metrics']
        plot_roc(eval_metrics, args)

    writer.close()



