import os
import sys
from urllib import request
import zipfile
import json
import math

import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import Dataset


class ChexpertSmall(Dataset):
    url = 'http://download.cs.stanford.edu/deep/CheXpert-v1.0-small.zip'
    dir_name = os.path.splitext(os.path.basename(url))[0]  # folder to match the filename
    attr_all_names = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
                      'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
                      'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
                      'Fracture', 'Support Devices']
    # select only the competition labels
    attr_names = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']

    def __init__(self, root, mode='train', transform=None, data_filter=None, mini_data=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        assert mode in ['train', 'valid', 'test', 'vis']
        self.mode = mode

        # if mode is test; root is path to csv file (in test mode), construct dataset from this csv;
        # if mode is train/valid; root is path to data folder with `train`/`valid` csv file to construct dataset.
        if mode == 'test':
            self.data = pd.read_csv(self.root, keep_default_na=True)
            self.root = '.'  # base path; to be joined to filename in csv file in __getitem__
        else:
            self._maybe_download_and_extract()
            self._maybe_process(data_filter)

            data_file = os.path.join(self.root, self.dir_name, 'valid.pt' if mode in ['valid', 'vis'] else 'train.pt')
            self.data = torch.load(data_file)

            if mini_data is not None:
                # truncate data to only a subset for debugging
                self.data = self.data[:mini_data]

            if mode=='vis':
                # select a subset of the data to visualize:
                #   3 examples from each condition category + no finding category + multiple conditions category

                # 1. get data idxs with a/ only 1 condition, b/ no findings, c/ 2 conditions, d/ >2 conditions; return list of lists
                idxs = []
                data = self.data
                for attr in self.attr_names:                                                               # 1 only condition category
                    idxs.append(self.data.loc[(self.data[attr]==1) & (self.data[self.attr_names].sum(1)==1), self.attr_names].head(3).index.tolist())
                idxs.append(self.data.loc[self.data[self.attr_names].sum(1)==0, self.attr_names].head(3).index.tolist())  # no findings category
                idxs.append(self.data.loc[self.data[self.attr_names].sum(1)==2, self.attr_names].head(3).index.tolist())  # 2 conditions category
                idxs.append(self.data.loc[self.data[self.attr_names].sum(1)>2, self.attr_names].head(3).index.tolist())   # >2 conditions category
                # save labels to visualize with a list of list of the idxs corresponding to each attribute
                self.vis_attrs = self.attr_names + ['No findings', '2 conditions', 'Multiple conditions']
                self.vis_idxs = idxs

                # 2. select only subset
                idxs_flatten = torch.tensor([i for sublist in idxs for i in sublist])
                self.data = self.data.iloc[idxs_flatten]

    def __getitem__(self, idx):
        # 1. select and load image
        img_path = self.data['Path'].iloc[idx]
        img = Image.open(os.path.join(self.root, img_path))
        if self.transform is not None:
            img = self.transform(img)

        # 2. select attributes as targets
        # NOTE attr is a vector of 0s under the test
        attr = torch.zeros(len(self.attr_names))
        if self.mode != 'test':
            attr = self.data[self.attr_names].iloc[idx].values
            attr = torch.tensor(attr).float()

        # 3. save index for extracting the patient_id in prediction/eval results as 'CheXpert-v1.0-small/valid/patient64541/study1'
        #    performed using the extract_patient_ids function
        idx = torch.tensor(self.data.index[idx])  # idx is based on len(self.data); if we are taking a subset of the data, idx will be relative to len(subset);
                                                  # self.data.index(idx) pull the index in the original dataframe and not the subset

        return img, attr, idx

    def __len__(self):
        return len(self.data)

    def _maybe_download_and_extract(self):
        fpath = os.path.join(self.root, os.path.basename(self.url))
        # if data dir does not exist, download file to root and unzip into dir_name
        if not os.path.exists(os.path.join(self.root, self.dir_name)):
            # check if zip file already downloaded
            if not os.path.exists(os.path.join(self.root, os.path.basename(self.url))):
                print('Downloading ' + self.url + ' to ' + fpath)
                def _progress(count, block_size, total_size):
                    sys.stdout.write('\r>> Downloading %s %.1f%%' % (fpath,
                        float(count * block_size) / float(total_size) * 100.0))
                    sys.stdout.flush()
                request.urlretrieve(self.url, fpath, _progress)
                print()
            print('Extracting ' + fpath)
            with zipfile.ZipFile(fpath, 'r') as z:
                z.extractall(self.root)
                if os.path.exists(os.path.join(self.root, self.dir_name, '__MACOSX')):
                    os.rmdir(os.path.join(self.root, self.dir_name, '__MACOSX'))
            os.unlink(fpath)
            print('Dataset extracted.')

    def _maybe_process(self, data_filter):
        # Dataset labels are: blank for unmentioned, 0 for negative, -1 for uncertain, and 1 for positive.
        # Process by:
        #    1. fill NAs (blanks for unmentioned) as 0 (negatives)
        #    2. fill -1 as 1 (U-Ones method described in paper)  # TODO -- setup options for uncertain labels
        #    3. apply attr filters as a dictionary {data_attribute: value_to_keep} e.g. {'Frontal/Lateral': 'Frontal'}

        # check for processed .pt files
        train_file = os.path.join(self.root, self.dir_name, 'train.pt')
        valid_file = os.path.join(self.root, self.dir_name, 'valid.pt')
        if not (os.path.exists(train_file) and os.path.exists(valid_file)):
            # load data and preprocess training data
            valid_df = pd.read_csv(os.path.join(self.root, self.dir_name, 'valid.csv'), keep_default_na=True)
            train_df = self._load_and_preprocess_training_data(os.path.join(self.root, self.dir_name, 'train.csv'), data_filter)

            # save
            torch.save(train_df, train_file)
            torch.save(valid_df, valid_file)

    def _load_and_preprocess_training_data(self, csv_path, data_filter):
        train_df = pd.read_csv(csv_path, keep_default_na=True)

        # 1. fill NAs (blanks for unmentioned) as 0 (negatives)
        # attr columns ['No Finding', ..., 'Support Devices']; note AP/PA remains with NAs for Lateral pictures
        train_df[self.attr_names] = train_df[self.attr_names].fillna(0)

        # 2. fill -1 as 1 (U-Ones method described in paper)  # TODO -- setup options for uncertain labels
        train_df[self.attr_names] = train_df[self.attr_names].replace(-1,1)

        if data_filter is not None:
            # 3. apply attr filters
            # only keep data matching the attribute e.g. df['Frontal/Lateral']=='Frontal'
            for k, v in data_filter.items():
                train_df = train_df[train_df[k]==v]

            with open(os.path.join(os.path.dirname(csv_path), 'processed_training_data_filters.json'), 'w') as f:
                json.dump(data_filter, f)

        return train_df


def extract_patient_ids(dataset, idxs):
    # extract a list of patient_id for prediction/eval results as ['CheXpert-v1.0-small/valid/patient64541/study1', ...]
    #    extract from image path = 'CheXpert-v1.0-small/valid/patient64541/study1/view1_frontal.jpg'
    #    NOTE -- patient_id is non-unique as there can be multiple views under the same study
    return dataset.data['Path'].loc[idxs].str.rsplit('/', expand=True, n=1)[0].values


def compute_mean_and_std(dataset):
    m = 0
    s = 0
    k = 1
    for img, _, _ in tqdm(dataset):
        x = img.mean().item()
        new_m = m + (x - m)/k
        s += (x - m)*(x - new_m)
        m = new_m
        k += 1
    print('Number of datapoints: ', k)
    return m, math.sqrt(s/(k-1))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='Data directory.')
    args = parser.parse_args()

    ds = ChexpertSmall(root=args.data_dir, mode='train')
    print('Train dataset loaded. Length: ', len(ds))

    output_dir = 'results/test/'

    # output a few images from the validation set and display labels
    if True:
        import torchvision.transforms as T
        from torchvision.utils import save_image
        ds = ChexpertSmall(root=args.data_dir, mode='valid',
                transform=T.Compose([T.CenterCrop(320), T.ToTensor(), T.Normalize(mean=[0.5330], std=[0.0349])]))
        print('Valid dataset loaded. Length: ', len(ds))
        for i in range(10):
            img, attr, patient_id = ds[i]
            save_image(img, 'test_valid_dataset_image_{}.png'.format(i), normalize=True, scale_each=True)
            print('Patient id: {}; labels: {}'.format(patient_id, attr))

    if False:
        ds = ChexpertSmall(root=args.data_dir, mode='train', transform=T.Compose([T.CenterCrop(320), T.ToTensor()]))
        m, s = compute_mean_and_std(ds)
        print('Dataset mean: {}; dataset std {}'.format(m, s))
        # Dataset mean: 0.533048452958796; dataset std 0.03490651403764978
