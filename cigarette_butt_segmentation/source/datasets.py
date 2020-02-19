#   Создание датасета, включает считывание исходных данных
#   с диска, их предобработку, аугментацию и перемешивание
import cv2
import json
import numpy as np
import pandas as pd
from PIL import Image

from pathlib import Path

import os
#os.chdir("..")
#import sys
#sys.path.append("..") 


from lib import *

import torch


from torch.utils.data import Dataset, DataLoader

DATA_MODES = ['train', 'val', 'test']

class CigaretteButtDataset(Dataset):

    def __init__(self,
                 img_dpath,
                 img_fnames,
                 img_transform,
                 mask_encodings=None,
                 mask_size=None,
                 mask_transform=None):
        self.img_dpath = img_dpath
        self.img_fnames = img_fnames
        self.img_transform = img_transform

        self.mask_encodings = mask_encodings
        self.mask_size = mask_size
        self.mask_transform = mask_transform

    def __getitem__(self, i):

        seed = np.random.randint(2147483647)

        fname = self.img_fnames[i]
        fpath = os.path.join(self.img_dpath, fname)
        img = Image.open(fpath)
        if self.img_transform is not None:
            random.seed(seed)
            img = self.img_transform(img)

        if self.mask_encodings is None:
            return img, fname

        if self.mask_size is None or self.mask_transform is None:
            raise ValueError('If mask_dpath is not None, mask_size and mask_transform must not be None.')

        mask = np.zeros(self.mask_size, dtype=np.uint8)
        if self.mask_encodings[fname][0] == self.mask_encodings[fname][0]: # NaN doesn't equal to itself
            for encoding in self.mask_encodings[fname]:
                mask += rle_decode(encoding, self.mask_size)
        mask = np.clip(mask, 0, 1)

        mask = Image.fromarray(mask)

        random.seed(seed)
        mask = self.mask_transform(mask)

        return img, torch.from_numpy(np.array(mask, dtype=np.int64))

    def __len__(self):
        return len(self.img_fnames)

def prepare_datasets():

    TRAIN_DIR = Path('data/train/images/')
    VAL_DIR = Path('data/val/images/')
    TEST_DIR = Path('data/real_test/')

    #path = "data/train"
    train_images = os.listdir(f"{path}/images")
    val_images = os.listdir()
    train_annotations = json.load(open(f"{TRAIN_DIR}/coco_annotations.json", "r"))
    #img_id = int(np.random.choice(images).split(".")[0])

    img = np.array(Image.open(f"{path}/images/{img_id:08}.jpg"))
    mask = get_mask(img_id, annotations)
    show_img_with_mask(img, mask)

    train_images = sorted(list(TRAIN_DIR.rglob('*.jpg')))
    val_files = sorted(list(VAL_DIR.rglob('*.jpg')))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    data_loader =  {'train': train_loader, 'val': val_loader, 'test': test_loader}

    return data_loader
