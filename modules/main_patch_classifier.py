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
from datasets import PatchClassifierDataset, create_dataloaders

from patch_classifier import PatchClassifier
from train_patch_classifier import kfold_patch_classifier, mean_kfold, kfold_boxplot, kfold_patient_classifier



DIRECTORY_ANNOTATED = "../HelicoDataSet/CrossValidation/Annotated"
PATH_PATCH_DIAGNOSIS = "../HelicoDataSet/HP_WSI-CoordAllAnnotatedPatches.xlsx"

PATH_PATIENT_DIAGNOSIS = "../HelicoDataSet/PatientDiagnosis.csv"

PATH_AUTOENCODER_WEIGHTS = "../trained_full/modelo_config1.pth"

BATCH_SIZE = 16

CONFIG = '1'

PATH_LOAD_PICKLE_DATASET = ""
PATH_SAVE_PICKLE_DATASET = ""

PATH_LOAD_PICKLE_CLASSIFIER_CALCULATIONS = ""
PATH_SAVE_PICKLE_CLASSIFIER_CALCULATIONS = ""

FOLDS = 5

THRESHOLD = 1.0009469407703822

def main():
    """
    Execute main logic of program.

    Returns
    -------
    None.

    """
    warnings.filterwarnings("ignore")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    autoencoder = AutoEncoderCNN(*AEConfigs(CONFIG))
    autoencoder.load_state_dict(torch.load(
        PATH_AUTOENCODER_WEIGHTS, map_location=device))

    model = PatchClassifier(autoencoder, device, 1)

    if PATH_LOAD_PICKLE_CLASSIFIER_CALCULATIONS != "":
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
        """PATCH KFOLD"""
        t0 = time.time()
        train_metrics, test_metrics = kfold_patch_classifier(model, dataset, device, BATCH_SIZE, FOLDS, show_roc=True)
        mean_train_thr, mean_train_fpr, mean_train_tpr = mean_kfold(train_metrics)
        tf = time.time()-t0
        print(f"Tiempo de ejecución: {tf:.4f} segundos")
        
        kfold_boxplot(train_metrics, "Threshold", "train_metrics")
        mean_test_acc, mean_test_fpr, mean_test_tpr = mean_kfold(test_metrics)
        kfold_boxplot(test_metrics, "Accuracy", "test_metrics")
    
        model.threshold = mean_train_thr
        
        """PATIENT KFOLD"""
        t0 = time.time()
        csv_patient_diagnosis = pd.read_csv(PATH_PATIENT_DIAGNOSIS)
        csv_patient_diagnosis["DENSITAT"][csv_patient_diagnosis["DENSITAT"] == "ALTA"] = 1
        csv_patient_diagnosis["DENSITAT"][csv_patient_diagnosis["DENSITAT"] == "BAIXA"] = 1
        csv_patient_diagnosis["DENSITAT"][csv_patient_diagnosis["DENSITAT"] == "NEGATIVA"] = 0
        
        train_metrics, test_metrics = kfold_patient_classifier(model, dataset, device, csv_patient_diagnosis, BATCH_SIZE, FOLDS, show_roc=True)

        mean_train_thr, mean_train_fpr, mean_train_tpr = mean_kfold(train_metrics)
        tf = time.time()-t0
        print(f"Tiempo de ejecución: {tf:.4f} segundos")
        
        kfold_boxplot(train_metrics, "Threshold", "train_metrics")
        mean_test_acc, mean_test_fpr, mean_test_tpr = mean_kfold(test_metrics)
        kfold_boxplot(test_metrics, "Accuracy", "test_metrics")

        print(mean_train_thr, mean_train_fpr, mean_train_tpr)
        print(mean_test_acc, mean_test_fpr, mean_test_tpr)
        
        
        
        


    # if PATH_SAVE_PICKLE_CLASSIFIER_CALCULATIONS != "":
    #     with open(PATH_SAVE_PICKLE_CLASSIFIER_CALCULATIONS, "wb") as file:
    #         data = {
    #             "fred_list": fred_list,
    #             "target_labels": target_labels,
    #         }

    #         pickle.dump(data, file)


if __name__ == "__main__":
    main()

