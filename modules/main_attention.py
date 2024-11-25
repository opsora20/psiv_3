import gc
import torch




import torch.optim as optim
from datasets import AutoEncoderDataset, create_dataloaders, PatchClassifierDataset, PatientDataset
from autoencoder import AEConfigs, AutoEncoderCNN
from train_autoencoder import train_attention
from utils import echo
import torchvision.models as models
from statistics_1 import Study_embeddings
import pandas as pd
from AttentionUnits import Attention, GatedAttention, NeuralNetwork, AttConfigs
from torch.nn import BCELoss
import os


DIRECTORY_CROPPED = "../HelicoDataSet/CrossValidation/Cropped"
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
    echo('Reading Dataset...')

    dataset = PatientDataset(
            PATH_PATIENT_DIAGNOSIS,
            DIRECTORY_CROPPED,
    )


    batch_size = 16
    num_epochs = 10


    loss_func = BCELoss() #Sigmoid loss

    dataloader = {}

    dataloader['train'] = create_dataloaders(dataset, batch_size)

    attConfig = 1

    output_size = [1000]
    optimizers = {}

    for config in range(1, 5):
        # CONFIG
        config = str(config)
        echo(f'Config: {config}')
        
        model_encoder = AutoEncoderCNN(*AEConfigs(config))
        model_encoder.to(device)

        netparamsAtt, netparamsNN = AttConfigs(attConfig, output_size)

        model_att = Attention(netparamsAtt)
        model_att.to(device)
        model = NeuralNetwork(netparamsNN)
        model.to(device)
        optimizers["attention"] = optim.Adam(model_att.parameters(), lr = 0.001)
        optimizers["NN"] = optim.Adam(model.parameters(), lr=0.001)
        model_att, model = train_attention(model_encoder, model_att, model, loss_func, device, dataset, dataloader, patient_labels, optimizers, num_epochs)       
        torch.save(
            model_att.state_dict(),
            os.path.join(
                DIRECTORY_SAVE_MODELS,
                "modelo_config" + config + "_att.pth",
            ),
        )
        
        torch.save(
            model.state_dict(),
            os.path.join(
                DIRECTORY_SAVE_MODELS,
                "modelo_config" + config + "_NN.pth",
            ),
        )
        # Free GPU Memory After Training
        gc.collect()
        torch.cuda.empty_cache()





if __name__ == "__main__":
    main()

