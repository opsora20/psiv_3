# -*- coding: utf-8 -*- noqa
"""
Created on Tue Nov 12 12:44:33 2024

@author: sergio
"""

import pickle
import torch
import warnings
import time
import pandas as pd

from autoencoder import AEConfigs, AutoEncoderCNN
from datasets import PatchClassifierDataset, create_dataloaders, PatientDataset

from patch_classifier import PatchClassifier
from train_patch_classifier import mean_kfold, kfold_boxplot, kfold_classifier, compute_all_cropped_fred
from utils import load_patient_diagnosis, save_pickle, load_pickle


DIRECTORY_CROPPED = "../HelicoDataSet/CrossValidation/Cropped"
DIRECTORY_ANNOTATED = "../HelicoDataSet/CrossValidation/Annotated"
PATH_PATCH_DIAGNOSIS = "../HelicoDataSet/HP_WSI-CoordAllAnnotatedPatches.xlsx"

PATH_PATIENT_DIAGNOSIS = "../HelicoDataSet/PatientDiagnosis.csv"

PATH_AUTOENCODER_WEIGHTS = ["../new_trained_full/modelo_config1.pth", "../new_trained_full/modelo_config2.pth",
                            "../new_trained_full/modelo_config3.pth"]

BATCH_SIZE = 16

CONFIG = ['1', '2', '3']

PATH_LOAD_PICKLE_FRED_CROPPED = ["../pickle_saves/cropped_fred_dict_new_model_1_full.pkl",
                                 "../pickle_saves/cropped_fred_dict_new_model_2_full.pkl",
                                 "../pickle_saves/cropped_fred_dict_new_model_3_full.pkl"]
PATH_SAVE_PICKLE_FRED_CROPPED = ["../pickle_saves/cropped_fred_dict_new_model_1_full.pkl",
                                 "../pickle_saves/cropped_fred_dict_new_model_2_full.pkl",
                                 "../pickle_saves/cropped_fred_dict_new_model_3_full.pkl"]

FOLDS = 5

def main():
    """
    Execute main logic of program.

    Returns
    -------
    None.

    """
    warnings.filterwarnings("ignore")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #for i in range(3):
    autoencoder = AutoEncoderCNN(*AEConfigs(CONFIG[2]))
    autoencoder.load_state_dict(torch.load(
        PATH_AUTOENCODER_WEIGHTS[2], map_location=device))

    model = PatchClassifier(autoencoder, device, 1)


    dataset_annotated = PatchClassifierDataset(
        PATH_PATCH_DIAGNOSIS,
        DIRECTORY_ANNOTATED,
        pickle_load_file='',
        pickle_save_file='',
    )

    df_patient_diagnosis = load_patient_diagnosis(PATH_PATIENT_DIAGNOSIS)
    
    
    #if PATH_SAVE_PICKLE_FRED_CROPPED != "":
        
    dataset_patient_cropped = PatientDataset(
        PATH_PATIENT_DIAGNOSIS,
        DIRECTORY_CROPPED
    )
    
    patients_fred_dict = compute_all_cropped_fred(model, dataset_patient_cropped, device, df_patient_diagnosis,
                                                    BATCH_SIZE, show_fred=False)
    save_pickle(patients_fred_dict, PATH_SAVE_PICKLE_FRED_CROPPED[2])
        
        
    #elif PATH_LOAD_PICKLE_FRED_CROPPED != "":
        
    patients_fred_dict = load_pickle(PATH_LOAD_PICKLE_FRED_CROPPED[2])
    
    diccionario_filtrado = {codi: valores for codi, valores in patients_fred_dict.items() if valores is not None}

    df_filtrado = df_patient_diagnosis[df_patient_diagnosis['CODI'].isin(diccionario_filtrado.keys())]
    
    kfold_classifier(model, dataset_annotated, device, df_filtrado, patients_fred_dict, BATCH_SIZE, FOLDS, CONFIG[2],
                        show_fred=False, show_roc_patch=True, show_roc_patient=True)

if __name__ == "__main__":
    main()

