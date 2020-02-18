#   Создание датасета, включает считывание исходных данных
#   с диска, их предобработку, аугментацию и перемешивание

import os

import cv2
import json
import numpy as np
import pandas as pd
from PIL import Image

from pathlib import Path

from lib import *

import torch


from torch.utils.data import Dataset, DataLoader

DATA_MODES = ['train', 'val', 'test']

class CigaretteButtDataset(Dataset):

    def __init__(self, file, mode):
        super().__init__()
        self.files = sorted(files)
        self.mode = mode

        if self.mode not in DATA_MODES:
            print(f"{self.mode} is not correct; correct modes: {DATA_MODES}")
            raise NameError

        self.len_ = len(self.files)

    def __len__(self):
        return self.len_


TRAIN_DIR = Path('../data/train/images/')
VAL_DIR = Path('../data/val/images/')
TEST_DIR = Path('../data/real_test/')

#path = "data/train"
train_images = os.listdir(f"{path}/images")
val_images = os.listdir()
annotations = json.load(open(f"{TRAIN_DIR}/coco_annotations.json", "r"))
#img_id = int(np.random.choice(images).split(".")[0])

img = np.array(Image.open(f"{path}/images/{img_id:08}.jpg"))
mask = get_mask(img_id, annotations)
show_img_with_mask(img, mask)

train_images = sorted(list(TRAIN_DIR.rglob('*.jpg')))
val_files = sorted(list(VAL_DIR.rglob('*.jpg')))
