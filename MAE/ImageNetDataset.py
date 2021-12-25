from __future__ import print_function

import torch
import os
from PIL import Image
import numpy as np


class MyDataSet:
    def __init__(self, root, transform=None, mask_ratio=0.75, used_ratio=0.4):
        self.transform = transform
        self.prefix = root
        self.imgs = []
        self.mask_ratio = mask_ratio
        self.num_patch = 14 * 14
        self.folders = os.listdir(self.prefix)

        for folder in self.folders:
            cur_folder_path = os.path.join(self.prefix, folder)
            if os.path.isdir(cur_folder_path):
                imgs = os.listdir(cur_folder_path)
                imgs = [img for img in imgs if img.endswith(".JPEG")]
                imgs = list(map(lambda x: os.path.join(cur_folder_path, x), imgs))
                self.imgs.extend(imgs)
            else:
                print("{} is not a vaild folder......".format(cur_folder_path))
        np.random.shuffle(self.imgs)

        self.length = len(self.imgs)
        valid_length = int(self.length * used_ratio)
        self.imgs = self.imgs[: valid_length]
        self.length = valid_length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        imgPath = self.imgs[index]
        image = Image.open(imgPath).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        mask = self.generate_mask()
        return image, mask

    def generate_mask(self):
        mask_num = int(self.num_patch * self.mask_ratio)

        mask = np.concatenate([np.zeros(self.num_patch - mask_num, dtype=np.bool), np.ones(mask_num, dtype=np.bool)], axis=0)
        np.random.shuffle(mask)
        return mask


class DataPrefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_input = next(self.loader)
        except StopIteration:
            self.next_input = None
            return
        with torch.cuda.stream(self.stream):
            batch, mask = self.next_input
            batch = batch.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            self.next_input = (batch, mask)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        inputs = self.next_input
        self.preload()
        return inputs
