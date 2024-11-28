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
import numpy as np
from AttentionUnits import Attention_NN, AttConfigs
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
import os
from utils import save_pickle, load_pickle
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision.models as models
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import KFold
from train_patch_classifier import dist_thr, get_best_thr


DIRECTORY_ANNOTATED = "../HelicoDataSet/CrossValidation/Annotated"
PATH_PATCH_DIAGNOSIS = "../HelicoDataSet/HP_WSI-CoordAllAnnotatedPatches.xlsx"



DIRECTORY_SAVE_MODELS = "../New_models"
PATH_PATIENT_DIAGNOSIS = "../HelicoDataSet/PatientDiagnosis.csv"

PATH_ATTENTION_WEIGHTS = "../New_models/modelo_config_resnet_att1.pth"


load_embeddings = False
is_resnet = True
num_epochs = 100

def main():
    csv_patient_diagnosis = pd.read_csv(PATH_PATIENT_DIAGNOSIS)
    csv_patient_diagnosis["DENSITAT"] = csv_patient_diagnosis["DENSITAT"].replace({
    "ALTA": 1,
    "BAIXA": 1,
    "NEGATIVA": 0
    })
    patient_labels = csv_patient_diagnosis.set_index('CODI').apply(lambda row: {'label': row['DENSITAT']}, axis=1).to_dict()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    echo(device)
    echo('Reading Dataset...')


    dataset = PatientDataset(
            PATH_PATIENT_DIAGNOSIS,
            DIRECTORY_ANNOTATED,
    )
    batch_size = 16
    config = '1'
    dataloader = {}
    dataloader['train'] = create_dataloaders(dataset, batch_size)
    
    model_encoder = models.resnet50(pretrained = True)
    model_encoder.to(device)

    output_size = [1000]

    if(load_embeddings):
        patient_labels = get_all_embeddings(patient_labels, dataset, model_encoder, output_size, device, dataloader, 100000, is_resnet)
        if(is_resnet):
            save_pickle(patient_labels, "Embeddings_dict_resnet_annotated")
        else:
            save_pickle(patient_labels, "Embeddings_dict_config"+str(config)+"_"+str(output_size[0])+"_annotated")
    else:
        if(is_resnet):
            patient_labels = load_pickle("Embeddings_dict_resnet")
        else:
            patient_labels = load_pickle("Embeddings_dict_config"+str(config)+"_"+str(output_size[0])+"_annotated")
    # Free GPU Memory After Training
    gc.collect()
    torch.cuda.empty_cache()

    attConfig = 1
    netparamsAtt, netparamsNN = AttConfigs(attConfig, output_size)

    model = Attention_NN(netparamsAtt, netparamsNN, gated = False)

    model.to(device)

    kfold_attention(patient_labels, netparamsAtt, netparamsNN, model, device)



def kfold_attention(patient_dict, netparamsAtt, netparamsNN, model, device):
    keys = list(patient_dict.keys())
    labels = np.array([patient_dict[k]["label"] for k in keys])
    
    k = 4
    kf = KFold(n_splits=k, shuffle=True)

    metrics = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(keys)):
        print(f"Fold {fold + 1}:")

        model = Attention_NN(netparamsAtt, netparamsNN, gated = False)

        model.to(device)
        # Dividir pacientes en entrenamiento y prueba
        train_keys = [keys[i] for i in train_idx]
        test_keys = [keys[i] for i in test_idx]

        train_data = {key: patient_dict[key] for key in train_keys}
        test_data = {key: patient_dict[key] for key in test_keys}

        loss_func = BCEWithLogitsLoss() #Sigmoid loss
        optimizer = optim.Adam(model.parameters(), lr = 0.001)

        model, loss_log = train_attention(model, loss_func, device, train_data, optimizer, num_epochs)
        plot_loss(loss_log)

        fpr, tpr, roc_auc, thr = test_attention(test_data, model, device)

        best_thr, best_fpr, best_tpr = get_best_thr(fpr, tpr, thr)

        metrics.append((best_thr, best_fpr, best_tpr))
    
    kfold_boxplot(metrics, "Att_boxplots.png")



def kfold_boxplot(metrics: list[tuple[float]], file_name: str):
    metric_1, metric_2, metric_3 = zip(*metrics)
    print(metric_1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 1 fila, 3 columnas

    # Crear boxplots individuales
    axes[0].boxplot(metric_1, patch_artist=True)
    axes[0].set_title("Thresholds")
    axes[0].set_ylabel('Values')

    axes[1].boxplot(metric_2, patch_artist=True)
    axes[1].set_title('False Positive Rate')

    axes[2].boxplot(metric_3, patch_artist=True)
    axes[2].set_title('True Positive Rate')

    # Ajustar el espacio entre subplots
    plt.tight_layout()

    plt.savefig(file_name+'.png')


def plot_loss(loss_log):
    losses = loss_log["train"]
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(losses) + 1), losses, marker='o', label="Training Loss", color='b')

    # Etiquetas y título
    plt.title("Training Loss Over Epochs", fontsize=16)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)

    # Mostrar el gráfico
    plt.show()

def test_attention(patient_dict, model, device):
    target = []
    preds = []
    model.eval()
    for patient, info in patient_dict.items():
        if (info["label"] == 1):
            target.append(1)
        else:
            target.append(0)
        patches = info["patches"].to(device)
        patient_preds = model(patches)
        patient_preds = patient_preds.reshape(2)
        preds.append(patient_preds[1].cpu().detach().numpy())
    
    fpr, tpr, thresholds = roc_curve(np.array(target), np.array(preds))

    roc_auc = roc_auc_score(target, preds)

    # Graficar la curva ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Línea diagonal
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

    return fpr, tpr, roc_auc, thresholds

if __name__ == "__main__":
    main()
