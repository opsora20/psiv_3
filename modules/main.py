# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 13:31:01 2024

@author: joanb
"""

import os
import torch
import torch.optim as optim
from torch.nn import MSELoss
import pandas as pd
# from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from cropped import load_cropped_patients, create_dataloaders
from Datasets import HelicoDataset
from models import AutoEncoderCNN
import gc
from train import train_autoencoder
from losses import MSE_loss

from utils import echo


ROOT_DIR = "../HelicoDataSet/CrossValidation/Cropped"
csv_filename = "../HelicoDataSet/PatientDiagnosis.csv"


def AEConfigs(Config):
    net_paramsEnc = {}
    net_paramsDec = {}
    inputmodule_paramsEnc = {}
    inputmodule_paramsEnc['num_input_channels'] = 3
    inputmodule_paramsDec = {}
    if Config == '1':
        # CONFIG1
        net_paramsEnc['block_configs'] = [[32, 32], [64, 64]]
        net_paramsEnc['stride'] = [[1, 2], [1, 2]]
        net_paramsDec['block_configs'] = [[inputmodule_paramsEnc['num_input_channels'], inputmodule_paramsEnc['num_input_channels']], [32,32]]
        net_paramsDec['stride'] = net_paramsEnc['stride']
        inputmodule_paramsDec['num_input_channels'] = net_paramsEnc['block_configs'][-1][-1]

        net_paramsEnc['dim'] = 2
        net_paramsDec['dim'] = 2

        net_paramsEnc['drop_rate'] = 0
        net_paramsDec['drop_rate'] = 0

    elif Config == '2':
        # CONFIG 2
        net_paramsEnc['block_configs'] = [[32], [64], [128], [256]]
        net_paramsEnc['stride'] = [[2], [2], [2], [2]]
        net_paramsDec['block_configs'] = [[128], [64], [32],
                                          [inputmodule_paramsEnc['num_input_channels']]]
        net_paramsDec['stride'] = net_paramsEnc['stride']
        inputmodule_paramsDec['num_input_channels'] = net_paramsEnc['block_configs'][-1][-1]

    elif Config == '3':
        # CONFIG3
        net_paramsEnc['block_configs'] = [[32], [64], [64]]
        net_paramsEnc['stride'] = [[1], [2], [2]]
        net_paramsDec['block_configs'] = [[64], [32], [
            inputmodule_paramsEnc['num_input_channels']]]
        net_paramsDec['stride'] = net_paramsEnc['stride']
        inputmodule_paramsDec['num_input_channels'] = net_paramsEnc['block_configs'][-1][-1]

    return net_paramsEnc, net_paramsDec, inputmodule_paramsDec, inputmodule_paramsEnc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    echo(device)
    Config = '1'
    net_paramsEnc, net_paramsDec, inputmodule_paramsDec, inputmodule_paramsEnc = AEConfigs(
        Config)
    model = AutoEncoderCNN(inputmodule_paramsEnc, net_paramsEnc,
                           inputmodule_paramsDec, net_paramsDec)
    echo(model)
    model.to(device)

    echo('Reading Dataset...')
    data = HelicoDataset(csv_filename, ROOT_DIR, read_images=True)
    batch_size = 16
    dataloader = create_dataloaders(data, batch_size)
    echo('Dataset Readed')
    # 0. EXPERIMENT PARAMETERS
    # 0.1 AE PARAMETERS

    # 0.1 NETWORK TRAINING PARAMS

    # 0.2 FOLDERS

    # 1. LOAD DATA
    # 1.1 Patient Diagnosis

    # 1.2 Patches Data

    # 2. DATA SPLITING INTO INDEPENDENT SETS

    # 2.0 Annotated set for FRed optimal threshold

    # 2.1 AE trainnig set

    # 2.1 Diagosis crossvalidation set

    # 3. lOAD PATCHES

    # 4. AE TRAINING

    # EXPERIMENTAL DESIGN:
    # TRAIN ON AE PATIENTS AN AUTOENCODER, USE THE ANNOTATED PATIENTS TO SET THE
    # THRESHOLD ON FRED, VALIDATE FRED FOR DIAGNOSIS ON A 10 FOLD SCHEME OF REMAINING
    # CASES.

    # 4.1 Data Split

    # CONFIG1

    # 4.2 Model Training
    loader = {}
    loader["train"] = dataloader
    loss_func = MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10

    train_autoencoder(model, batch_size, loss_func, device,
                      loader, optimizer, num_epochs)

    # Free GPU Memory After Training
    gc.collect()
    torch.cuda.empty_cache()
    # 5. AE RED METRICS THRESHOLD LEARNING

    # 5.1 AE Model Evaluation

    # Free GPU Memory After Evaluation
    gc.collect()
    torch.cuda.empty_cache()

    # 5.2 RedMetrics Threshold

    # 6. DIAGNOSIS CROSSVALIDATION
    # 6.1 Load Patches 4 CrossValidation of Diagnosis

    # 6.2 Diagnostic Power


if __name__ == "__main__":
    main()
