# -*- coding: utf-8 -*- noqa
"""
Created on Tue Nov 12 12:44:33 2024

@author: sergio
"""

import pickle
import torch
import warnings

from autoencoder import AEConfigs, AutoEncoderCNN
from datasets import PatchClassifierDataset, create_dataloaders

from patch_classifier import PatchClassifier
from train_patch_classifier import train_patch_classifier
from roc_curve import roc


DIRECTORY_ANNOTATED = "../HelicoDataSet/CrossValidation/Annotated"
PATH_PATCH_DIAGNOSIS = "../HelicoDataSet/HP_WSI-CoordAllAnnotatedPatches.xlsx"

PATH_AUTOENCODER_WEIGHTS = "../trained_full/modelo_config1.pth"

BATCH_SIZE = 16

CONFIG = '1'

PATH_LOAD_PICKLE_DATASET = ""
PATH_SAVE_PICKLE_DATASET = ""

PATH_LOAD_PICKLE_CLASSIFIER_CALCULATIONS = ""
PATH_SAVE_PICKLE_CLASSIFIER_CALCULATIONS = ""


def main():
    """
    Execute main logic of program.

    Returns
    -------
    None.

    """
    warnings.filterwarnings("ignore")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    autoencoder = AutoEncoderCNN(AEConfigs(CONFIG))
    autoencoder.load_state_dict(torch.load(
        PATH_AUTOENCODER_WEIGHTS, map_location=device))

    model = PatchClassifier(autoencoder, device, 1)

    if PATH_LOAD_PICKLE_CLASSIFIER_CALCULATIONS == "":
        with open(PATH_LOAD_PICKLE_CLASSIFIER_CALCULATIONS, "rb") as file:
            data = pickle.load(file)

            fred_list = data["fred_list"]
            target_labels = data["target_labels"]

    else:
        dataset = PatchClassifierDataset(
            PATH_PATCH_DIAGNOSIS,
            DIRECTORY_ANNOTATED,
            pickle_load_file=PATH_LOAD_PICKLE_DATASET,
            pickle_save_file=PATH_SAVE_PICKLE_DATASET,
        )

        dataloader = create_dataloaders(dataset, BATCH_SIZE)

        fred_list, target_labels = train_patch_classifier(model, dataloader)

    if PATH_SAVE_PICKLE_CLASSIFIER_CALCULATIONS != "":
        with open(PATH_SAVE_PICKLE_CLASSIFIER_CALCULATIONS, "wb") as file:
            data = {
                "fred_list": fred_list,
                "target_labels": target_labels,
            }

            pickle.dump(data, file)

    roc(
        target_labels,
        fred_list,
        f"ROC Curve for HelicoBacter in patches [{CONFIG}]",
    )


if __name__ == "__main__":
    main()
