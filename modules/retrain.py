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
import pickle
from utils import echo


ROOT_DIR = "../../CrossValidation/Cropped"
csv_filename = "../HelicoDataSet/PatientDiagnosis.csv"
PATH_AEMODEL = "../Local_Train"


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
        net_paramsDec['block_configs'] = [[inputmodule_paramsEnc['num_input_channels'],
                                           inputmodule_paramsEnc['num_input_channels']], [32, 32]]
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
        net_paramsDec['block_configs'] = [
            [inputmodule_paramsEnc['num_input_channels']], [32], [64], [128]]
        net_paramsDec['stride'] = net_paramsEnc['stride']
        inputmodule_paramsDec['num_input_channels'] = net_paramsEnc['block_configs'][-1][-1]

        net_paramsEnc['dim'] = 2
        net_paramsDec['dim'] = 2

        net_paramsEnc['drop_rate'] = 0
        net_paramsDec['drop_rate'] = 0

    elif Config == '3':
        # CONFIG3
        net_paramsEnc['block_configs'] = [[32], [64], [64]]
        net_paramsEnc['stride'] = [[1], [2], [2]]
        net_paramsDec['block_configs'] = [
            [inputmodule_paramsEnc['num_input_channels']], [32], [64]]
        net_paramsDec['stride'] = net_paramsEnc['stride']
        inputmodule_paramsDec['num_input_channels'] = net_paramsEnc['block_configs'][-1][-1]

        net_paramsEnc['dim'] = 2
        net_paramsDec['dim'] = 2

        net_paramsEnc['drop_rate'] = 0
        net_paramsDec['drop_rate'] = 0

    elif Config == '4':
        # CONFIG1
        net_paramsEnc['block_configs'] = [[32, 32], [64, 64], [128, 128]]
        net_paramsEnc['stride'] = [[1, 2], [1, 2], [1, 2]]
        net_paramsDec['block_configs'] = [[inputmodule_paramsEnc['num_input_channels'],
                                           inputmodule_paramsEnc['num_input_channels']], [32, 32], [64, 64]]
        net_paramsDec['stride'] = net_paramsEnc['stride']
        inputmodule_paramsDec['num_input_channels'] = net_paramsEnc['block_configs'][-1][-1]

        net_paramsEnc['dim'] = 2
        net_paramsDec['dim'] = 2

        net_paramsEnc['drop_rate'] = 0
        net_paramsDec['drop_rate'] = 0

    return net_paramsEnc, net_paramsDec, inputmodule_paramsDec, inputmodule_paramsEnc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    echo(device)
    for i in range(3, 6):
        echo('Reading Dataset...'+ROOT_DIR+"_"+str(i))
        data = HelicoDataset(csv_filename, ROOT_DIR+"_"+str(i), read_images=True)
        batch_size = 16
        dataloader = create_dataloaders(data, batch_size)
        echo('Dataset Readed')

        for Config in range(1, 5):
            Config = str(Config)
            echo(f'Config: {Config}')
            net_paramsEnc, net_paramsDec, inputmodule_paramsDec, inputmodule_paramsEnc = AEConfigs(
                Config)
            model = AutoEncoderCNN(inputmodule_paramsEnc, net_paramsEnc,
                                inputmodule_paramsDec, net_paramsDec)
            
            model.load_state_dict(torch.load(os.path.join(PATH_AEMODEL, "modelo_config"+str(Config)+".pth"), map_location=device))
            echo(model)
            model.to(device)

            loader = {}
            loader["train"] = dataloader
            loss_func = MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            num_epochs = 10

            autoencoder = train_autoencoder(model, batch_size, loss_func, device,
                                            loader, optimizer, num_epochs)

            torch.save(autoencoder.state_dict(),
                    os.path.join("..", "Local_Train", "modelo_config"+str(Config)+".pth"))
            gc.collect()
            torch.cuda.empty_cache()
        del data



if __name__ == "__main__":
    main()
