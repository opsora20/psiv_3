# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 13:31:01 2024

@author: joanb
"""

import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from cropped import load_cropped_patients, create_dataloaders
from Datasets import HelicoDataset


ROOT_DIR = "../HelicoDataSet/CrossValidation/Cropped"
csv_filename = "../HelicoDataSet/PatientDiagnosis.csv"

if "__name__" == "__main__":
    data = HelicoDataset(csv_filename, ROOT_DIR)
    batch_size = 16
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
    dataloader = create_dataloaders(data, batch_size)
    