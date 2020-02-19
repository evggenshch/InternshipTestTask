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

    def load_sample(self, file):
        image = Image.open(file)
        image.load()
        return image

    def __getitem__(self, index):
        # для преобразования изображений в тензоры PyTorch и нормализации входа
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        x = self.load_sample(self.files[index])
        x = self._prepare_sample(x)
        x = np.array(x / 255, dtype='float32')
        x = transform(x)
        if self.mode == 'test':
            return x
        else:
            label = self.labels[index]
            label_id = self.label_encoder.transform([label])
            y = label_id.item()
            return x, y

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
