import gc
import torch
import os

import torch.optim as optim

from torch.nn import MSELoss

from datasets import AutoEncoderDataset, create_dataloaders, PatchClassifierDataset
from autoencoder import AEConfigs, AutoEncoderCNN
from train_autoencoder import train_autoencoder
from utils import echo
import torchvision.models as models
from statistics_1 import Study_embeddings

DIRECTORY_ANNOTATED = "../HelicoDataSet/CrossValidation/Annotated"
PATH_PATCH_DIAGNOSIS = "../HelicoDataSet/HP_WSI-CoordAllAnnotatedPatches.xlsx"

DIRECTORY_SAVE_MODELS = "../trained_full"

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
    model = models.resnet50(pretrained = True)
    model.to(device)

    echo('Reading Dataset...')

    dataset = PatchClassifierDataset(
            PATH_PATCH_DIAGNOSIS,
            DIRECTORY_ANNOTATED,
            pickle_load_file=PATH_LOAD_PICKLE_DATASET,
            pickle_save_file=PATH_SAVE_PICKLE_DATASET,
    )


    batch_size = 16

    dataloader = create_dataloaders(dataset, batch_size)

    echo('Dataset Readed')

    #emb_class = Study_embeddings(dataloader, model, device, is_resnet = True)
    #emb_class.plot_embeddings()

    for config in range(1, 5):
        # CONFIG
        config = str(config)
        echo(f'Config: {config}')
        
        model = AutoEncoderCNN(*AEConfigs(config))

        print(model.encoder)
        model.to(device)

        emb_class = Study_embeddings(dataloader, model, device, is_resnet = False)
        emb_class.plot_embeddings()

        # Free GPU Memory After Training
        gc.collect()
        torch.cuda.empty_cache()





if __name__ == "__main__":
    main()

