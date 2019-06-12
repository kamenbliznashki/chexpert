import os
import argparse
import json

import torch
import pandas as pd

from chexpert import fetch_dataloader, load_json, densenet121, resnet152
from dataset import ChexpertSmall, extract_patient_ids


# TODO  1. fix defaults for:
#           a/ restore_path to codalab `src/<path>` convention for ensemble or single model
#           b/ model
#           c/ resize


parser = argparse.ArgumentParser()
parser.add_argument('data_path', type=str, help='Path to input data csv file.')
parser.add_argument('output_path', type=str, help='Path for output csv file (e.g. /predictions.csv).')
# model params
parser.add_argument('--restore_path', type=str, help='Path to a single model checkpoint to restore or path to folder of checkpoints to ensemble.')
parser.add_argument('--model', default='densenet121', choices=['densenet121', 'resnet152'], help='What model architecture to use.')
parser.add_argument('--cuda', type=int, help='Which cuda device to use.')
# dataloader params
parser.add_argument('--batch_size', type=int, default=16, help='Dataloader batch size.')
parser.add_argument('--resize', type=int, help='Size of minimum edge to which to resize images.')
parser.add_argument('--mini_data', type=int, help='Truncate dataset to first entry only.')
# testing
parser.add_argument('--debug', action='store_true', help='Evaluate prediction output against dataseta single model.')


@torch.no_grad()
def predict(model, dataloader, args):
    model.eval()

    probs, patient_ids = [], []
    for x, _, idx in dataloader:
        scores = model(x.to(args.device))

        probs += [scores.sigmoid().cpu()]
        patient_ids += extract_patient_ids(dataloader.dataset, idxs)

    probs = torch.cat(probs, 0).numpy()

    # pull unique patient_ids 'CheXpert-v1.0-small/valid/patient64541/study1'
    # and take max of probabilities per class over the multiple views
    df = pd.DataFrame(data=probs, index=patient_ids, columns=[*dataloader.dataset.attr_names])
    df.index.name = 'Study'
    df = df.groupby('Study').max()

    return df


if __name__ == '__main__':
    args = parser.parse_args()
    args.device = torch.device('cuda:{}'.format(args.cuda) if args.cuda is not None and torch.cuda.is_available() else 'cpu')

    # load model and weights
    model = locals()[args.model](num_classes=len(ChexpertSmall.attr_names)).to(args.device)

    # if folder is provided to restore, run ensemble prediction of every checkpoint in the folder, else run single model
    args.ensemble = os.path.isdir(args.restore_path)

    # load pretrained from config -- in case args.pretrained flag is forgotten e.g. in post-training evaluation
    # (images still need to be normalized if training started on an imagenet pretrained model)
    args.pretrained = load_json(os.path.join(os.path.dirname(args.restore_path), 'config.json'))['pretrained']

    # load data
    dataloader = fetch_dataloader(args, mode='test')

    # run predictions and save results
    if args.ensemble:
        # get checkpoint paths
        checkpoints = [c for c in os.listdir(args.restore_path) if c.startswith('checkpoint') and c.endswith('.pt')]
        print('Running ensemble prediction using {} checkpoints.'.format(len(checkpoints)))

        dfs = []
        for checkpoint in checkpoints:
            # load state dict
            model_checkpoint = torch.load(os.path.join(args.restore_path, checkpoint), map_location=args.device)
            model.load_state_dict(model_checkpoint['state_dict'])
            # run prediction
            dfs += [predict(model, dataloader, args)]

        # concat results and mean
        df = pd.concat(dfs, axis=1).groupby(axis=1, level=0).mean()  # concat over columns and mean over checkpoints
    else:
        print('Running prediction using {}'.format(args.restore_path))
        # load state dict
        model_checkpoint = torch.load(args.restore_path, map_location=args.device)
        model.load_state_dict(model_checkpoint['state_dict'])
        # run prediction
        df = predict(model, dataloader, args)

    # write results to csv
    df.to_csv(args.output_path)


    # test -- provide data_path as valid.csv dataset and run predictions above against validation targets
    if args.debug:
        args.data_path = ''
        mode = 'valid'
        dataloader = fetch_dataloader(args, mode)
        targets, patient_ids = [], []
        for _, target, idxs in dataloader:
            targets += [target]
            patient_ids += extract_patient_ids(dataloader.dataset, idxs)
        targets = pd.DataFrame(data=torch.cat(targets, 0).numpy(), index=patient_ids, columns=[*dataloader.dataset.attr_names])
        targets.index.name = 'Study'
        targets = targets.groupby('Study').max()

        from chexpert import compute_metrics
        metrics = compute_metrics(torch.from_numpy(df.values), torch.from_numpy(targets.values), torch.zeros(1, len(dataloader.dataset.attr_names)))
        print('Metrics for predictions vs targets:\n\tdataset mode: {}\n\trestore_path: {}'.format(mode, args.restore_path))
        print('AUC:\n', metrics['aucs'])
