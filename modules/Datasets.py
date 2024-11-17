# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 12:44:33 2024

@author: joanb
"""

import os
import torch
import pandas as pd
# from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from cropped import load_cropped_patients, load_annotated_patients


class HelicoDataset(Dataset):

    def __init__(self, csv_file, root_dir, read_images=False, transform=None, cropped=True):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self._csv = pd.read_csv(csv_file)
        self._root_dir = root_dir
        self._transform = transform
        self._cropped = cropped
        self._labels = np.array

        if (read_images):
            self.__read_images()

    def __read_images(self):
        if self._cropped:
            self._images = load_cropped_patients(self._root_dir, self._csv)
        else:
            self._images, self._patient, self._labels = load_annotated_patients(
                self._root_dir, self._excel)

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        if self._cropped:
            if torch.is_tensor(idx):
                idx = idx.tolist()
            sample = self._images[idx]
            if (self._transform):
                sample = self._transform(sample)
            sample = sample.astype(np.float32)
            return torch.from_numpy(sample)
        else:
            if torch.is_tensor(idx):
                idx = idx.tolist()
            img_sample = self._images[idx]
            label_sample = self._labels[idx]
            if (self._transform):
                img_sample = self._transform(img_sample)
            img_sample = img_sample.astype(np.float32)
            return torch.from_numpy(img_sample), torch.from_numpy(label_sample)
