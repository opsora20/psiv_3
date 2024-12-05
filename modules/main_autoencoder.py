# -*- coding: utf-8 -*- noqa
"""
Created on Tue Nov 12 13:31:01 2024

@author: joanb
"""

import gc
import torch
import os

import matplotlib.pyplot as plt

import torch.optim as optim

from torch.nn import MSELoss

from datasets import AutoEncoderDataset, create_dataloaders
from autoencoder import AEConfigs, AutoEncoderCNN
from train_autoencoder import train_autoencoder
from utils import echo


DIRECTORY_CROPPED = os.path.join(
    "..", "..", "maed", "HelicoDataSet", "CrossValidation", "Cropped")
PATH_PATIENT_DIAGNOSIS = "../../maed/HelicoDataSet/PatientDiagnosis.csv"

DIRECTORY_SAVE_MODELS = "../models"

PATH_SAVE_PICKLE_DATASET = "../sd.pckl"
PATH_LOAD_PICKLE_DATASET = "../sd.pckl"

def plot_multiple_losses(loss_logs):
    """
    Plotea múltiples curvas de pérdida a partir de una lista de diccionarios.
    
    Parameters:
    - loss_logs: list[dict]
        Una lista de diccionarios, donde cada diccionario tiene un formato {"name": str, "train": list}
        "name" es el nombre de la serie (e.g., modelo, experimento).
        "train" es una lista de valores de pérdida por época.
    """
    plt.figure(figsize=(10, 7))
    config = [1, 3]
    i = 0
    for log in loss_logs:
        name = log.get("name", "Config: "+str(config[i]))
        losses = log.get("train", [])  # Lista de pérdidas.
        plt.plot(range(1, len(losses) + 1), losses, marker='o', label=name)
        i+=1

    # Etiquetas y título
    plt.title("Training Loss AutoEncoder", fontsize=16)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)

    # Mostrar el gráfico
    plt.savefig("losses.png")


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
        read=True,
        pickle_load_file=PATH_LOAD_PICKLE_DATASET,
        pickle_save_file=PATH_SAVE_PICKLE_DATASET,
    )

    batch_size = 256

    dataloader = create_dataloaders(data, batch_size)

    echo('Dataset Readed')
    
    losses = []

    for config in [1, 3]:
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
        num_epochs = 150

        autoencoder, loss_log = train_autoencoder(
            model,
            loss_func,
            device,
            loader,
            optimizer,
            num_epochs,
            config,
            DIRECTORY_SAVE_MODELS,
        )
        
        losses.append(loss_log)

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
        
    plot_multiple_losses(losses)


if __name__ == "__main__":
    main()
