import gc
import torch



from datasets import AutoEncoderDataset, create_dataloaders, PatchClassifierDataset
from autoencoder import AEConfigs, AutoEncoderCNN
from train_autoencoder import train_autoencoder
from utils import echo
import torchvision.models as models
from statistics_1 import Study_embeddings
import pandas as pd
from AttentionUnits import Attention, GatedAttention
from torch.nn import BCELoss


DIRECTORY_ANNOTATED = "../HelicoDataSet/CrossValidation/Annotated"
PATH_PATCH_DIAGNOSIS = "../HelicoDataSet/HP_WSI-CoordAllAnnotatedPatches.xlsx"

DIRECTORY_SAVE_MODELS = "../trained_full"
PATH_PATIENT_DIAGNOSIS = "../HelicoDataSet/PatientDiagnosis.csv"
PATH_SAVE_PICKLE_DATASET = ""
PATH_LOAD_PICKLE_DATASET = ""


def main():
    """
    Execute main logic of program.

    Returns
    -------
    None.

    """
    csv_patient_diagnosis = pd.read_csv(PATH_PATIENT_DIAGNOSIS)
    csv_patient_diagnosis["DENSITAT"][csv_patient_diagnosis["DENSITAT"] == "ALTA"] = 1
    csv_patient_diagnosis["DENSITAT"][csv_patient_diagnosis["DENSITAT"] == "BAIXA"] = 1
    csv_patient_diagnosis["DENSITAT"][csv_patient_diagnosis["DENSITAT"] == "NEGATIVA"] = 0
    
    patient_labels  = csv_patient_diagnosis.set_index('CODI')['DENSITAT'].to_dict()

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


    loss_Func = BCELoss() #Sigmoid loss

    echo('Dataset Readed')

    for config in range(1, 5):
        # CONFIG
        config = str(config)
        echo(f'Config: {config}')
        
        model_encoder = AutoEncoderCNN(*AEConfigs(config))


        model.to(device)



        # Free GPU Memory After Training
        gc.collect()
        torch.cuda.empty_cache()





if __name__ == "__main__":
    main()

