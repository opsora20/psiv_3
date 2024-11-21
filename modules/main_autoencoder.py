# -*- coding: utf-8 -*- noqa
"""
Created on Tue Nov 12 13:31:01 2024

@author: joanb
"""

import gc
import torch
import os

import torch.optim as optim

from torch.nn import MSELoss

from datasets import AutoEncoderDataset, create_dataloaders
from autoencoder import AEConfigs, AutoEncoderCNN
from train_autoencoder import train_autoencoder
from utils import echo


DIRECTORY_CROPPED = "../HelicoDataSet/CrossValidation/Cropped"
PATH_PATIENT_DIAGNOSIS = "../HelicoDataSet/PatientDiagnosis.csv"

DIRECTORY_SAVE_MODELS = "../models"

PATH_SAVE_PICKLE_DATASET = ""
PATH_LOAD_PICKLE_DATASET = ""


def main():
    """
    Execute main logic of program.

    Returns
    -------
    None.

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    echo(device)

    echo('Reading Dataset...')

    data = AutoEncoderDataset(
        PATH_PATIENT_DIAGNOSIS,
        DIRECTORY_CROPPED,
        pickle_load_file=PATH_LOAD_PICKLE_DATASET,
        pickle_save_file=PATH_SAVE_PICKLE_DATASET,
    )

    batch_size = 16

    dataloader = create_dataloaders(data, batch_size)

    echo('Dataset Readed')

    for config in range(1, 5):
        # CONFIG
        config = str(config)
        echo(f'Config: {config}')

        model = AutoEncoderCNN(*AEConfigs(config))
        echo(model)

        model.to(device)

        # Model Training
        loader = {}
        loader["train"] = dataloader
        loss_func = MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        num_epochs = 10

        autoencoder = train_autoencoder(
            model,
            loss_func,
            device,
            loader,
            optimizer,
            num_epochs
        )

        torch.save(
            autoencoder.state_dict(),
            os.path.join(
                DIRECTORY_SAVE_MODELS,
                "modelo_config" + config + ".pth",
            ),
        )

        # Free GPU Memory After Training
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
