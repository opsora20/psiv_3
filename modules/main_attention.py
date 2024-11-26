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
from torch.nn import BCEWithLogitsLoss
import os
from utils import save_pickle, load_pickle


DIRECTORY_CROPPED = "../HelicoDataSet/CrossValidation/Cropped"
PATH_PATCH_DIAGNOSIS = "../HelicoDataSet/HP_WSI-CoordAllAnnotatedPatches.xlsx"

DIRECTORY_SAVE_MODELS = "../trained_full"
PATH_PATIENT_DIAGNOSIS = "../HelicoDataSet/PatientDiagnosis.csv"
PATH_SAVE_PICKLE_DATASET = ""
PATH_LOAD_PICKLE_DATASET = ""

load_embeddings = False

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
    num_epochs = 10

    dataloader = {}

    dataloader['train'] = create_dataloaders(dataset, batch_size)

    attConfig = 1

    output_size = [1024]
    config = '1'

    model_encoder = AutoEncoderCNN(*AEConfigs(config))
    model_encoder.to(device)
    # Free GPU Memory After Training
    gc.collect()
    torch.cuda.empty_cache()
    if(load_embeddings):

        patient_labels = get_all_embeddings(patient_labels, dataset, model_encoder, output_size, device, dataloader, max_images=100000)
        save_pickle(patient_labels, "Embeddings_dict_config"+str(config))
    else:
        patient_labels = load_pickle("Embeddings_dict_config"+str(config))
    # Free GPU Memory After Training
    gc.collect()
    torch.cuda.empty_cache()

    netparamsAtt, netparamsNN = AttConfigs(attConfig, output_size)

    model = Attention_NN(netparamsAtt, netparamsNN)
    model.to(device)
    
    loss_func = BCEWithLogitsLoss() #Sigmoid loss
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    print(len(patient_labels))

    model = train_attention(model, loss_func, device, patient_labels, optimizer, num_epochs)

    torch.save(
            model.state_dict(),
            os.path.join(
                DIRECTORY_SAVE_MODELS,
                "modelo_config" + config + "_att.pth",
            ),
        )

    gc.collect()
    torch.cuda.empty_cache()

    """
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
        print(model)
        model_att, model = train_attention(model_encoder, model_att, model, loss_func, device, dataset, dataloader, patient_labels, output_size, optimizers, num_epochs)       
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
    """




if __name__ == "__main__":
    main()

