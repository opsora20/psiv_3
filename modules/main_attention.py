import gc
import torch




import torch.optim as optim
from datasets import AutoEncoderDataset, create_dataloaders, PatchClassifierDataset, PatientDataset
from autoencoder import AEConfigs, AutoEncoderCNN
from train_autoencoder import train_attention
from utils import echo, get_all_embeddings
import torchvision.models as models
from statistics_1 import Study_embeddings
import pandas as pd
from AttentionUnits import Attention_NN, AttConfigs
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
import os
from utils import save_pickle, load_pickle
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision.models as models


DIRECTORY_CROPPED = "../HelicoDataSet/CrossValidation/Cropped"
PATH_PATCH_DIAGNOSIS = "../HelicoDataSet/HP_WSI-CoordAllAnnotatedPatches.xlsx"

DIRECTORY_SAVE_MODELS = "../New_models"
PATH_PATIENT_DIAGNOSIS = "../HelicoDataSet/PatientDiagnosis.csv"
PATH_SAVE_PICKLE_DATASET = ""
PATH_LOAD_PICKLE_DATASET = ""

load_embeddings = False
is_resnet = True

def main():
    """
    Execute main logic of program.

    Returns
    -------
    None.

    """
    csv_patient_diagnosis = pd.read_csv(PATH_PATIENT_DIAGNOSIS)
    csv_patient_diagnosis["DENSITAT"] = csv_patient_diagnosis["DENSITAT"].replace({
    "ALTA": 1,
    "BAIXA": 1,
    "NEGATIVA": 0
    })

    # Crear el diccionario con la estructura solicitada
    patient_labels = csv_patient_diagnosis.set_index('CODI').apply(lambda row: {'label': row['DENSITAT']}, axis=1).to_dict()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    echo(device)
    echo('Reading Dataset...')

    dataset = PatientDataset(
            PATH_PATIENT_DIAGNOSIS,
            DIRECTORY_CROPPED,
    )


    batch_size = 16
    num_epochs = 100

    dataloader = {}

    dataloader['train'] = create_dataloaders(dataset, batch_size)

    attConfig = 1


    
    config = '1'

    if(is_resnet):
        model_encoder = models.resnet50(pretrained = True)
        output_size = [1000]
    else:
        model_encoder = AutoEncoderCNN(*AEConfigs(config))
        output_size = [1024]
    
    model_encoder.to(device)



    if(load_embeddings):

        patient_labels = get_all_embeddings(patient_labels, dataset, model_encoder, output_size, device, dataloader, 100000, is_resnet)
        if(is_resnet):
            save_pickle(patient_labels, "Embeddings_dict_resnet")
        else:
            save_pickle(patient_labels, "Embeddings_dict_config"+str(config)+"_"+str(output_size[0]))
    else:
        if(is_resnet):
            patient_labels = load_pickle("Embeddings_dict_resnet")
        else:
            patient_labels = load_pickle("Embeddings_dict_config"+str(config)+"_"+str(output_size[0]))
    # Free GPU Memory After Training
    gc.collect()
    torch.cuda.empty_cache()

    netparamsAtt, netparamsNN = AttConfigs(attConfig, output_size)

    model = Attention_NN(netparamsAtt, netparamsNN, gated = False)

    model.to(device)
    
    loss_func = BCEWithLogitsLoss() #Sigmoid loss
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    print(len(patient_labels))

    model = train_attention(model, loss_func, device, patient_labels, optimizer, num_epochs)

    torch.save(
            model.state_dict(),
            os.path.join(
                DIRECTORY_SAVE_MODELS,
                "modelo_config" + str(attConfig) + "_att.pth",
            ),
        )

    gc.collect()
    torch.cuda.empty_cache()



if __name__ == "__main__":
    main()

