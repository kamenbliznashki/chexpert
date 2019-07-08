import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader, TensorDataset

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
from functools import partial

# dataset and models
from dataset import ChexpertSmall, extract_patient_ids
from torchvision.models import densenet121, resnet152
from models.efficientnet import construct_model
from models.attn_aug_conv import DenseNet, ResNet, Bottleneck


parser = argparse.ArgumentParser()
# action
parser.add_argument('--load_config', type=str, help='Path to config.json file to load args from.')
parser.add_argument('--train', action='store_true', help='Train model.')
parser.add_argument('--evaluate_single_model', action='store_true', help='Evaluate a single model.')
parser.add_argument('--evaluate_ensemble', action='store_true', help='Evaluate an ensemble (given a checkpoints tracker of saved model checkpoints).')
parser.add_argument('--visualize', action='store_true', help='Visualize Grad-CAM.')
parser.add_argument('--plot_roc', action='store_true', help='Filename for metrics json file to plot ROC.')
parser.add_argument('--seed', type=int, default=0, help='Random seed to use.')
parser.add_argument('--cuda', type=int, help='Which cuda device to use.')
# paths
parser.add_argument('--data_path', default='', help='Location of train/valid datasets directory or path to test csv file.')
parser.add_argument('--output_dir', help='Path to experiment output, config, checkpoints, etc.')
parser.add_argument('--restore', type=str, help='Path to a single model checkpoint to restore or folder of checkpoints to ensemble.')
# model architecture
parser.add_argument('--model', default='densenet121', help='What model architecture to use. (densenet121, resnet152, efficientnet-b[0-7])')
# data params
parser.add_argument('--mini_data', type=int, help='Truncate dataset to this number of examples.')
parser.add_argument('--resize', type=int, help='Size of minimum edge to which to resize images.')
# training params
parser.add_argument('--pretrained', action='store_true', help='Use ImageNet pretrained model and normalize data mean and std.')
parser.add_argument('--batch_size', type=int, default=16, help='Dataloaders batch size.')
parser.add_argument('--n_epochs', type=int, default=1, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
parser.add_argument('--lr_warmup_steps', type=float, default=0, help='Linear warmup of the learning rate for lr_warmup_steps number of steps.')
parser.add_argument('--lr_decay_factor', type=float, default=0.97, help='Decay factor if exponential learning rate decay scheduler.')
parser.add_argument('--step', type=int, default=0, help='Current step of training (number of minibatches processed).')
parser.add_argument('--log_interval', type=int, default=50, help='Interval of num batches to show loss statistics.')
parser.add_argument('--eval_interval', type=int, default=300, help='Interval of num epochs to evaluate, checkpoint, and save samples.')


# --------------------
# Data IO
# --------------------

def fetch_dataloader(args, mode):
    assert mode in ['train', 'valid', 'vis']

    transforms = T.Compose([
        T.Resize(args.resize) if args.resize else T.Lambda(lambda x: x),
        T.CenterCrop(320),
        T.ToTensor(),
        T.Normalize(mean=[0.5330], std=[0.0349]),   # whiten with dataset mean and std
        T.Lambda(lambda x: x.expand(3,-1,-1))])     # expand to 3 channels

    dataset = ChexpertSmall(args.data_path, mode, transforms, mini_data=args.mini_data)

    return DataLoader(dataset, args.batch_size, shuffle=(mode=='train'), pin_memory=(args.device.type=='cuda'),
                      num_workers=0 if mode=='valid' else 4)  # since evaluating the valid_dataloader is called inside the
                                                              # train_dataloader loop, 0 workers for valid_dataloader avoids
                                                              # forking (cf torch dataloader docs); else memory sharing gets clunky

def save_json(data, filename, args):
    with open(os.path.join(args.output_dir, filename + '.json'), 'w') as f:
        json.dump(data, f, indent=4)

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def save_checkpoint(checkpoint, optim_checkpoint, sched_checkpoint, args, max_records=10):
    """ save model and optimizer checkpoint along with csv tracker
    of last `max_records` best number of checkpoints as sorted by avg auc """
    # 1. save latest
    torch.save(checkpoint, os.path.join(args.output_dir, 'checkpoint_latest.pt'))
    torch.save(optim_checkpoint, os.path.join(args.output_dir, 'optim_checkpoint_latest.pt'))
    if sched_checkpoint: torch.save(sched_checkpoint, os.path.join(args.output_dir, 'sched_checkpoint_latest.pt'))

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
        torch.save(checkpoint, os.path.join(args.output_dir, 'best_checkpoints', 'checkpoint_{}.pt'.format(file_id)))


# --------------------
# Evaluation metrics
# --------------------

def compute_metrics(outputs, targets, losses):
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

def train_epoch(model, train_dataloader, valid_dataloader, loss_fn, optimizer, scheduler, writer, epoch, args):
    model.train()

    with tqdm(total=len(train_dataloader), desc='Step at start {}; Training epoch {}/{}'.format(args.step, epoch+1, args.n_epochs)) as pbar:
        for x, target, idxs in train_dataloader:
            args.step += 1

            out = model(x.to(args.device))
            loss = loss_fn(out, target.to(args.device)).sum(1).mean(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler and args.step >= args.warmup_steps: scheduler.step()

            pbar.set_postfix(loss = '{:.4f}'.format(loss.item()))
            pbar.update()

            # record
            if args.step % args.log_interval == 0:
                writer.add_scalar('train_loss', loss.item(), args.step)
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], args.step)

            # evaluate and save on eval_interval
            if args.step % args.eval_interval == 0:
                with torch.no_grad():
                    model.eval()

                    eval_metrics = evaluate_single_model(model, valid_dataloader, loss_fn, args)

                    writer.add_scalar('eval_loss', np.sum(list(eval_metrics['loss'].values())), args.step)
                    for k, v in eval_metrics['aucs'].items():
                        writer.add_scalar('eval_auc_class_{}'.format(k), v, args.step)

                    # save model
                    save_checkpoint(checkpoint={'global_step': args.step,
                                                'eval_loss': np.sum(list(eval_metrics['loss'].values())),
                                                'avg_auc': np.nanmean(list(eval_metrics['aucs'].values())),
                                                'state_dict': model.state_dict()},
                                    optim_checkpoint=optimizer.state_dict(),
                                    sched_checkpoint=scheduler.state_dict() if scheduler else None,
                                    args=args)

                    # switch back to train mode
                    model.train()

@torch.no_grad()
def evaluate(model, dataloader, loss_fn, args):
    model.eval()

    targets, outputs, losses = [], [], []
    for x, target, idxs in dataloader:
        out = model(x.to(args.device))
        loss = loss_fn(out, target.to(args.device))

        outputs += [out.cpu()]
        targets += [target]
        losses  += [loss.cpu()]

    return torch.cat(outputs), torch.cat(targets), torch.cat(losses)

def evaluate_single_model(model, dataloader, loss_fn, args):
    outputs, targets, losses = evaluate(model, dataloader, loss_fn, args)
    return compute_metrics(outputs, targets, losses)

def evaluate_ensemble(model, dataloader, loss_fn, args):
    checkpoints = [c for c in os.listdir(args.restore) \
                        if c.startswith('checkpoint') and c.endswith('.pt')]
    print('Running ensemble prediction using {} checkpoints.'.format(len(checkpoints)))
    outputs, losses = [], []
    for checkpoint in checkpoints:
        # load weights
        model_checkpoint = torch.load(os.path.join(args.restore, checkpoint), map_location=args.device)
        model.load_state_dict(model_checkpoint['state_dict'])
        del model_checkpoint
        # evaluate
        outputs_, targets, losses_ = evaluate(model, dataloader, loss_fn, args)
        outputs += [outputs_]
        losses  += [losses_]

    # take mean over checkpoints
    outputs  = torch.stack(outputs, dim=2).mean(2)
    losses = torch.stack(losses, dim=2).mean(2)

    return compute_metrics(outputs, targets, losses)

def train_and_evaluate(model, train_dataloader, valid_dataloader, loss_fn, optimizer, scheduler, writer, args):
    for epoch in range(args.n_epochs):
        # train
        train_epoch(model, train_dataloader, valid_dataloader, loss_fn, optimizer, scheduler, writer, epoch, args)

        # evaluate
        print('Evaluating...', end='\r')
        eval_metrics = evaluate_single_model(model, valid_dataloader, loss_fn, args)
        print('Evaluate metrics @ step {}:'.format(args.step))
        print('AUC:\n', pprint.pformat(eval_metrics['aucs']))
        print('Loss:\n', pprint.pformat(eval_metrics['loss']))
        writer.add_scalar('eval_loss', np.sum(list(eval_metrics['loss'].values())), args.step)
        for k, v in eval_metrics['aucs'].items():
            writer.add_scalar('eval_auc_class_{}'.format(k), v, args.step)

        # save eval metrics
        save_json(eval_metrics, 'eval_results_step_{}'.format(args.step), args)

# --------------------
# Visualization
# --------------------

@torch.enable_grad()
def grad_cam(model, x, hooks, cls_idx=None):
    """ cf CheXpert: Test Results / Visualization; visualize final conv layer, using grads of final linear layer as weights,
    and performing a weighted sum of the final feature maps using those weights.
    cf Grad-CAM https://arxiv.org/pdf/1610.02391.pdf """

    model.eval()
    model.zero_grad()

    # register backward hooks
    conv_features, linear_grad = [], []
    forward_handle = hooks['forward'].register_forward_hook(lambda module, in_tensor, out_tensor: conv_features.append(out_tensor))
    backward_handle = hooks['backward'].register_backward_hook(lambda module, grad_input, grad_output: linear_grad.append(grad_input))

    # run model forward and create a one hot output for the given cls_idx or max class
    outputs = model(x)
    if not cls_idx: cls_idx = outputs.argmax(1)
    one_hot = F.one_hot(cls_idx, outputs.shape[1]).float().requires_grad_(True)

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

    # cleanup
    forward_handle.remove()
    backward_handle.remove()
    model.zero_grad()

    return cam

def visualize(model, dataloader, grad_cam_hooks, args):
    attr_names = dataloader.dataset.attr_names

    # 1. run through model to compute logits and grad-cam
    imgs, labels, scores, masks, idxs = [], [], [], [], []
    for x, target, idx in dataloader:
        imgs += [x]
        labels += [target]
        idxs += idx.tolist()
        x = x.to(args.device)
        scores += [model(x).cpu()]
        masks  += [grad_cam(model, x, grad_cam_hooks).cpu()]
    imgs, labels, scores, masks = torch.cat(imgs), torch.cat(labels), torch.cat(scores), torch.cat(masks)

    # 2. renormalize images and convert everything to numpy for matplotlib
    imgs.mul_(0.0349).add_(0.5330)
    imgs = imgs.permute(0,2,3,1).data.numpy()
    labels = labels.data.numpy()
    patient_ids = extract_patient_ids(dataloader.dataset, idxs)
    masks = masks.permute(0,2,3,1).data.numpy()
    probs = scores.sigmoid().data.numpy()

    # 3. make column grid of [model probs table, original image, grad-cam image] for each attr + other categories
    for attr, vis_idxs in zip(dataloader.dataset.vis_attrs, dataloader.dataset.vis_idxs):
        fig, axs = plt.subplots(3, 3, figsize=(4 * imgs.shape[1]/100, 3.3 * imgs.shape[2]/100), dpi=100, frameon=False)
        fig.suptitle(attr)
        for i, idx in enumerate(vis_idxs):
            offset = idxs.index(idx)
            visualize_one(model, imgs[offset], masks[offset], labels[offset], patient_ids[offset], probs[offset], attr_names, axs[i])

        filename = 'vis_{}_step_{}.png'.format(attr.replace(' ', '_'), args.step)
        plt.savefig(os.path.join(args.output_dir, 'vis', filename), dpi=100)
        plt.close()

def visualize_one(model, img, mask, label, patient_id, prob, attr_names, axs):
    """ display [table of model vs ground truth probs | original image | grad-cam mask image] in a given suplot axs """
    # sort data by prob high to low
    sort_idxs = prob.argsort()[::-1]
    label = label[sort_idxs]
    prob = prob[sort_idxs]
    names = [attr_names[i] for i in sort_idxs]
    # 1. left -- show table of ground truth and predictions, sorted by pred prob high to low
    axs[0].set_title(patient_id)
    data = np.stack([label, prob.round(3)]).T
    axs[0].table(cellText=data, rowLabels=names, colLabels=['Ground truth', 'Pred. prob'],
                 rowColours=plt.cm.Greens(0.5*label),
                 cellColours=plt.cm.Greens(0.5*data), cellLoc='center', loc='center')
    axs[0].axis('tight')
    # 2. middle -- show original image
    axs[1].set_title('Original image', fontsize=10)
    axs[1].imshow(img.squeeze(), cmap='gray')
    # 3. right -- show heatmap over original image with predictions
    axs[2].set_title('Top class activation \n{}: {:.4f}'.format(names[0], prob[0]), fontsize=10)
    axs[2].imshow(img.squeeze(), cmap='gray')
    axs[2].imshow(mask.squeeze(), cmap='jet', alpha=0.5)

    for ax in axs: ax.axis('off')

def vis_attn(x, patient_ids, idxs, attn_layers, args, batch_element=0):
    H, W = x.shape[2:]
    nh = attn_layers[0].nh

    # select which pixels to visualize -- e.g. select virtices of a center square of side 1/3 of the image dims
    pix_to_vis = lambda h, w: [(h//3, w//3), (h//3, int(2*w/3)), (int(2*h/3), w//3), (int(2*h/3), int(2*w/3))]
    window = 30  # take mean attn around the pix_to_vis in a window of size ws

    for j, l in enumerate(attn_layers):
        # visualize attention maps (rows for each head; columns for each pixel)
        fig, axs = plt.subplots(nh+1, 4, figsize=(3,3/4*(1+nh)), frameon=False)
        fig.suptitle(patient_ids[batch_element], fontsize=8)
        # display target image; highlight pixel
        for ax, (ph, pw) in zip(axs[0], pix_to_vis(H,W)):
            image = x.clone().detach().mul_(0.0349).add_(0.5330)  # renormalize
            image[:,:,ph-window:ph+window,pw-window:pw+window] = torch.tensor([1., 215/255, 0]).view(1,3,1,1)   # add yellow pixel on the pix_to_vis for visualization
            ax.imshow(image[batch_element].permute(1,2,0).numpy())
            ax.axis('off')
        # display attention maps
        # get attention weights tensor for the batch element
        attn = l.weights.data[batch_element]
        # reshape attn tensor and select the pixels to visualize
        h = w = int(np.sqrt(attn.shape[-1]))
        ws = max(1, int(window * h/H))  # scale window to feature map size
        attn = attn.reshape(nh, h, w, h, w)
        for i, (ph, pw) in enumerate(pix_to_vis(h,w)):
            for h in range(nh):
                axs[h+1, i].imshow(attn[h, ph-ws:ph+ws, pw-ws:pw+ws, :, :].mean([0,1]).cpu().numpy())
                axs[h+1, i].axis('off')


        filename = 'attn_image_idx_{}_{}_layer_{}.png'.format(idxs[batch_element], batch_element, j)
        fig.subplots_adjust(0,0,1,0.95,0.05,0.05)
        plt.savefig(os.path.join(args.output_dir, 'vis', filename))
        plt.close()

def plot_roc(metrics, args, filename, labels=ChexpertSmall.attr_names):
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

    plt.suptitle(filename)
    axs[0,0].set_ylabel('True Positive Rate')
    axs[1,0].set_ylabel('Precision')

    for ax in axs.flatten():
        ax.set_xlim([0.0, 1.05])
        ax.set_ylim([0.0, 1.05])
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'plots', filename + '.png'), pad_inches=0.)
    plt.close()

# --------------------
# Main
# --------------------

if __name__ == '__main__':
    args = parser.parse_args()

    # overwrite args from config
    if args.load_config: args.__dict__.update(load_json(args.load_config))

    # set up output folder
    if not args.output_dir:
        if args.restore: raise RuntimeError('Must specify `output_dir` argument')
        args.output_dir: args.output_dir = os.path.join('results', time.strftime('%Y-%m-%d_%H-%M-%S', time.gmtime()))
    # make new folders if they don't exist
    writer = SummaryWriter(logdir=args.output_dir)  # creates output_dir
    if not os.path.exists(os.path.join(args.output_dir, 'vis')): os.makedirs(os.path.join(args.output_dir, 'vis'))
    if not os.path.exists(os.path.join(args.output_dir, 'plots')): os.makedirs(os.path.join(args.output_dir, 'plots'))
    if not os.path.exists(os.path.join(args.output_dir, 'best_checkpoints')): os.makedirs(os.path.join(args.output_dir, 'best_checkpoints'))

    # save config
    if not os.path.exists(os.path.join(args.output_dir, 'config.json')): save_json(args.__dict__, 'config', args)
    writer.add_text('config', str(args.__dict__))

    args.device = torch.device('cuda:{}'.format(args.cuda) if args.cuda is not None and torch.cuda.is_available() else 'cpu')

    if args.seed:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # load model
    n_classes = len(ChexpertSmall.attr_names)
    if args.model=='densenet121':
        model = densenet121(pretrained=args.pretrained).to(args.device)
        # 1. replace output layer with chexpert number of classes (pretrained loads ImageNet n_classes)
        model.classifier = nn.Linear(model.classifier.in_features, out_features=n_classes).to(args.device)
        # 2. init output layer with default torchvision init
        nn.init.constant_(model.classifier.bias, 0)
        # 3. store locations of forward and backward hooks for grad-cam
        grad_cam_hooks = {'forward': model.features.norm5, 'backward': model.classifier}
        # 4. init optimizer and scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = None
#        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
#        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [40000, 60000])
    elif args.model=='aadensenet121':
        model = DenseNet(32, (6, 12, 24, 16), 64, num_classes=n_classes,
                         attn_params={'k': 0.2, 'v': 0.1, 'nh': 8, 'relative': True, 'input_dims': (320,320)}).to(args.device)
        grad_cam_hooks = {'forward': model.features, 'backward': model.classifier}
        attn_hooks = [model.features.transition1.conv, model.features.transition2.conv, model.features.transition3.conv]
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [40000, 60000])
    elif args.model=='resnet152':
        model = resnet152(pretrained=args.pretrained).to(args.device)
        model.fc = nn.Linear(model.fc.in_features, out_features=n_classes).to(args.device)
        grad_cam_hooks = {'forward': model.layer4, 'backward': model.fc}
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = None
    elif args.model=='aaresnet152':  # resnet50 layers [3,4,6,3]; resnet101 layers [3,4,23,3]; resnet 152 layers [3,8,36,3]
        model = ResNet(Bottleneck, [3, 8, 36, 3], num_classes=n_classes,
                         attn_params={'k': 0.2, 'v': 0.1, 'nh': 8, 'relative': True, 'input_dims': (320,320)}).to(args.device)
        grad_cam_hooks = {'forward': model.layer4, 'backward': model.fc}
        attn_hooks = [model.layer2[i].conv2 for i in range(len(model.layer2))] + \
                     [model.layer3[i].conv2 for i in range(len(model.layer3))] + \
                     [model.layer4[i].conv2 for i in range(len(model.layer4))]
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = None
    elif 'efficientnet' in args.model:
        model = construct_model(args.model, n_classes=n_classes).to(args.device)
        grad_cam_hooks = {'forward': model.head[1], 'backward': model.head[-1]}
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, momentum=0.9, eps=0.001)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.lr_decay_factor)
    else:
        raise RuntimeError('Model architecture not supported.')

    if args.restore and os.path.isfile(args.restore):  # restore from single file, else ensemble is handled by evaluate_ensemble
        print('Restoring model weights from {}'.format(args.restore))
        model_checkpoint = torch.load(args.restore, map_location=args.device)
        model.load_state_dict(model_checkpoint['state_dict'])
        args.step = model_checkpoint['global_step']
        del model_checkpoint
        # if training, load optimizer and scheduler too
        if args.train:
            print('Restoring optimizer.')
            optim_checkpoint_path = os.path.join(os.path.dirname(args.restore), 'optim_' + os.path.basename(args.restore))
            optimizer.load_state_dict(torch.load(optim_checkpoint_path, map_location=args.device))
            if scheduler:
                print('Restoring scheduler.')
                sched_checkpoint_path = os.path.join(os.path.dirname(args.restore), 'sched_' + os.path.basename(args.restore))
                scheduler.load_state_dict(torch.load(sched_checkpoint_path, map_location=args.device))

    # load data
    if args.restore:
        # load pretrained flag from config -- in case forgotten e.g. in post-training evaluation
        # (images still need to be normalized if training started on an imagenet pretrained model)
        args.pretrained = load_json(os.path.join(args.output_dir, 'config.json'))['pretrained']
    train_dataloader = fetch_dataloader(args, mode='train')
    valid_dataloader = fetch_dataloader(args, mode='valid')
    vis_dataloader = fetch_dataloader(args, mode='vis')

    # setup loss function for train and eval
    loss_fn = nn.BCEWithLogitsLoss(reduction='none').to(args.device)

    print('Loaded {} (number of parameters: {:,}; weights trained to step {})'.format(
        model._get_name(), sum(p.numel() for p in model.parameters()), args.step))
    print('Train data length: ', len(train_dataloader.dataset))
    print('Valid data length: ', len(valid_dataloader.dataset))
    print('Vis data subset: ', len(vis_dataloader.dataset))

    if args.train:
        train_and_evaluate(model, train_dataloader, valid_dataloader, loss_fn, optimizer, scheduler, writer, args)

    if args.evaluate_single_model:
        eval_metrics = evaluate_single_model(model, valid_dataloader, loss_fn, args)
        print('Evaluate metrics -- \n\t restore: {} \n\t step: {}:'.format(args.restore, args.step))
        print('AUC:\n', pprint.pformat(eval_metrics['aucs']))
        print('Loss:\n', pprint.pformat(eval_metrics['loss']))
        save_json(eval_metrics, 'eval_results_step_{}'.format(args.step), args)

    if args.evaluate_ensemble:
        assert os.path.isdir(args.restore), 'Restore argument must be directory with saved checkpoints'
        eval_metrics = evaluate_ensemble(model, valid_dataloader, loss_fn, args)
        print('Evaluate ensemble metrics -- \n\t checkpoints path {}:'.format(args.restore))
        print('AUC:\n', pprint.pformat(eval_metrics['aucs']))
        print('Loss:\n', pprint.pformat(eval_metrics['loss']))
        save_json(eval_metrics, 'eval_results_ensemble', args)

    if args.visualize:
        visualize(model, vis_dataloader, grad_cam_hooks, args)
        if attn_hooks is not None:
            for x, _, idxs in vis_dataloader:
                model(x.to(args.device))
                patient_ids = extract_patient_ids(vis_dataloader.dataset, idxs)
                # visualize stored attention weights for each image
                for i in range(len(x)): vis_attn(x, patient_ids, idxs, attn_hooks, args, i)

    if args.plot_roc:
        # load results files from output_dir
        filenames = [f for f in os.listdir(args.output_dir) if f.startswith('eval_results') and f.endswith('.json')]
        if filenames==[]: raise RuntimeError('No `eval_results` files found in `{}` to plot results from.'.format(args.output_dir))
        # load and plot each
        for f in filenames:
            plot_roc(load_json(os.path.join(args.output_dir, f)), args, 'roc_pr_' + f.split('.')[0])

    writer.close()
