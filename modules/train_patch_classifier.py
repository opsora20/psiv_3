# -*- coding: utf-8 -*- noqa
"""
Created on Wed Nov 20 23:06:19 2024

@author: JoelT
"""

from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import sys
from patch_classifier import PatchClassifier
from datasets import PatchClassifierDataset, PatientDataset
from torch import device
import torch
from collections import defaultdict
import pandas as pd
from utils import compare_histograms


def kfold_classifier(model: PatchClassifier, dataset: PatchClassifierDataset, device: device, df_filtrado: pd.DataFrame,
                           patient_fred_dict: dict, batch_size: int, k: int, config: str, show_fred=False, show_roc_patch=False,
                           show_roc_patient=False):
    patches = dataset.images
    patches = torch.from_numpy(patches).float()
    labels = dataset.labels
    labels[labels == -1] = 0
    patients = dataset.patients

    sgkf = StratifiedGroupKFold(n_splits=k)
    train_patch_metrics = []
    test_patch_metrics = []
    
    train_patient_metrics = []
    test_patient_metrics = []
    
    roc_curves_patches = []
    roc_curves_patients = []
    print("CONFIG: "+config+"\n\n")
    
    for fold, (train_index, test_index) in enumerate(sgkf.split(patches, labels, patients)):
        
        print("Fold:",fold)
        patches_train = patches[train_index]
        patches_test = patches[test_index]
        labels_train = labels[train_index]
        labels_test = labels[test_index]
        patients_train = np.unique(patients[train_index])
        patients_test = np.unique(patients[test_index])

        """PATCH THRESHOLD"""
        
        """TRAIN"""
        fred_list = np.array(compute_patches(model, device, patches_train, batch_size, show_fred, train=True)).astype(np.float64)
        fred_list = fred_list.astype(np.float64).clip(min=np.finfo(np.float64).min, max=np.finfo(np.float64).max)

        best_threshold_patch, train_patch_fpr, train_patch_tpr = compute_train_roc(fred_list, labels_train, str(fold)+"_patch", config, show_roc_patch)
        fpr, tpr, thr = roc_curve(labels_train, fred_list)
        roc_auc = auc(fpr, tpr)
        roc_curves_patches.append((fpr, tpr, roc_auc))
        
        
        print("Threshold",best_threshold_patch, "FPR", train_patch_fpr, "TPR", train_patch_tpr)
        model.threshold = best_threshold_patch
        predicted_labels = compute_patches(model, device, patches_test, batch_size, show_fred, test=True)

        """TEST"""
        test_patch_acc, test_patch_fpr, test_patch_tpr = obtain_test_metrics(predicted_labels, labels_test)
        print("Accuracy_patch",test_patch_acc, "FPR_patch", test_patch_fpr, "TPR_patch", test_patch_tpr)
        
        train_patch_metrics.append((best_threshold_patch, train_patch_fpr, train_patch_tpr))
        test_patch_metrics.append((test_patch_acc, test_patch_fpr, test_patch_tpr))
        
        """PATIENT THRESHOLD"""
        
        """TRAIN"""
        proporciones = {codi: sum(1 for x in valores if x > best_threshold_patch) / len(valores) 
                for codi, valores in patient_fred_dict.items() if valores is not None}

        # Crear una nueva columna en el DataFrame para las proporciones
        df_filtrado['PROPORTION'] = df_filtrado['CODI'].map(proporciones)
        
        df_train = df_filtrado[df_filtrado['CODI'].isin(patients_train)]
        densitats_train = list(df_train['DENSITAT'])
        proportions_train = list(df_train['PROPORTION'])
        
        best_threshold_patient, train_patient_fpr, train_patient_tpr = compute_train_roc(proportions_train, densitats_train, str(fold)+"_patient", config, show_roc_patient)
        fpr, tpr, thr = roc_curve(densitats_train, proportions_train)
        roc_auc = auc(fpr, tpr)
        roc_curves_patients.append((fpr, tpr, roc_auc))
        
        """TEST"""
        df_test = df_filtrado[df_filtrado['CODI'].isin(patients_test)]
        densitats_test = list(df_test['DENSITAT'])
        proportions_test = df_test['PROPORTION'].to_numpy()
        densitat_test_pred = list((proportions_test > best_threshold_patient).astype(int))
        test_patient_acc, test_patient_fpr, test_patient_tpr = obtain_test_metrics(densitats_test, densitat_test_pred)

        train_patient_metrics.append((best_threshold_patient, train_patient_fpr, train_patient_tpr))
        test_patient_metrics.append((test_patient_acc, test_patient_fpr, test_patient_tpr))       
    plot_all(train_patch_metrics, test_patch_metrics, train_patient_metrics, test_patient_metrics, config)
    plot_roc_curves(roc_curves_patches)
    plot_roc_curves(roc_curves_patients)

    return train_patch_metrics, test_patch_metrics, train_patient_metrics, test_patient_metrics

def plot_all(train_patch_metrics, test_patch_metrics, train_patient_metrics, test_patient_metrics, config):
    kfold_boxplot(train_patch_metrics, "Threshold", "train_patch_metrics", config)
    kfold_boxplot(test_patch_metrics, "Accuracy", "test_patch_metrics", config)
    kfold_boxplot(train_patient_metrics, "Threshold", "train_patient_metrics", config)
    kfold_boxplot(test_patient_metrics, "Accuracy", "test_patient_metrics", config)
    
    
def compute_all_cropped_fred(model: PatchClassifier, dataset: PatientDataset, device: device,
                            df_diagnosis: pd.DataFrame, batch_size: int, show_fred=False):
    patients_fred_dict = {codi: None for codi in df_diagnosis['CODI']}
    for patient in patients_fred_dict.keys():
        fred_list_patient = compute_patient_cropped_fred(model, device, dataset, patient, batch_size, show_fred)
        patients_fred_dict[patient] = fred_list_patient
    return patients_fred_dict


def compute_patient_cropped_fred(model: PatchClassifier, device: device, dataset: PatientDataset,
                                patient: str, batch_size: int, show_fred):
    exists = dataset.load_patient(patient, max_images=10_000)
    if exists:
        patches = dataset.images
        patches = torch.from_numpy(patches).float()
        fred_list = compute_patches(model, device, patches, batch_size, show_fred, train=True)
        print(patient)
        print(fred_list)
        return fred_list
    return
    
    
def compute_patches(model: PatchClassifier, device: device, patches: np.ndarray, batch_size: int,
                    show_fred, train=False, test=False):
    
    # histogramas_in = []
    # histogramas_out = []
    
    size = patches.shape[0]
    if train:
        fred_list = []   
    elif test:
        predicted_label = []
    
    pos = 0
    last = batch_size
    while pos < size:
        if last > size:
            last = size
        inputs_batch = patches[pos : last]
        inputs_batch = inputs_batch.to(device)
        outputs_batch = model.encode(inputs_batch)
        
        for input_image, output_image in zip(inputs_batch, outputs_batch):
            if train:
                fred_result = model.calculate_fred(
                input_image,
                output_image,
                show_fred,
        )
                fred_list.append(fred_result)
                
                # histogramas_in.append(in_hist)
                # histogramas_out.append(out_hist)
                
            if test:
                decided_class = model.execute(
                input_image,
                output_image
        )       
                predicted_label.append(decided_class)    
            
        pos += batch_size
        last += batch_size
    # histogramas_in = np.array(histogramas_in).astype(np.float64)
    # histogramas_in = np.mean(histogramas_in, axis=0)
    # histogramas_out = np.array(histogramas_out).astype(np.float64)
    # histogramas_out = np.mean(histogramas_out, axis=0)
    # compare_histograms(histogramas_in, histogramas_out, bin_edges)

    if train:
        return fred_list

    elif test:
        return predicted_label
        

def obtain_test_metrics(target_labels: list[int], predicted_labels: list[int]):
    conf_matrix = confusion_matrix(target_labels, predicted_labels)
    tn, fp, fn, tp = conf_matrix.ravel()
    acc = (tp + tn) / (tp + fp + tn + fn)
    fpr = fp / (fp + tn)
    tpr = tp / (tp + fn)
     
    return acc, fpr, tpr
        
    
def compute_train_roc(fred_list: list[float], target_labels: list[int], name, config, show = False):
    fpr, tpr, thr = roc_curve(target_labels, fred_list)
    if show:
        plot_roc(fpr, tpr, name, config)
    best_threshold, best_fpr, best_tpr = get_best_thr(fpr, tpr, thr)
    
    return best_threshold, best_fpr, best_tpr


def plot_roc(false_positive_rates: np.ndarray, true_positive_rates: np.ndarray, name, config):
    roc_auc = auc(false_positive_rates, true_positive_rates)
    # Plot the ROC curve
    plt.figure()  
    plt.plot(false_positive_rates, true_positive_rates, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for H. Pylori Patch Classification')
    plt.legend()
    plt.savefig("Roc_curve_fold_"+str(name)+"_config_"+config+".png")


def get_best_thr(false_positive_rates: np.ndarray, true_positive_rates: np.ndarray, thresholds: np.ndarray) -> tuple[float, float, float]:
    best_thr = None
    min_distance = sys.maxsize
    for fpr, tpr, thr in zip(false_positive_rates, true_positive_rates, thresholds):
        dist = dist_thr(fpr, tpr)
        if dist < min_distance:
            min_distance = dist
            best_thr= thr
            best_fpr = fpr
            best_tpr = tpr
    return best_thr, best_fpr, best_tpr
        
        
def dist_thr(false_positive_rate: float, true_positive_rate: float) -> float:
    dist = pow(pow(false_positive_rate, 2) + pow(1-true_positive_rate, 2), 0.5)
    return dist


def mean_kfold(metrics: list[tuple[float]]) -> list[tuple[float, float]]:
    metric_1, metric_2, metric_3 = zip(*metrics)
    metrics = [metric_1, metric_2, metric_3]
    metrics_stats = []
    for metric in metrics:
        print(metric)
        media = sum(metric) / len(metric)

        # Varianza
        varianza = sum((x - media) ** 2 for x in metric) / len(metric)

        # Desviación estándar poblacional
        desviacion = varianza ** 0.5
        metrics_stats.append((media, desviacion))
    return metrics_stats
    

def kfold_boxplot(metrics: list[tuple[float]], title_1: str, file_name: str, config: str):
    metric_1, metric_2, metric_3 = zip(*metrics)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 1 fila, 3 columnas

    # Crear boxplots individuales
    axes[0].boxplot(metric_1, patch_artist=True)
    axes[0].set_title(title_1)
    axes[0].set_ylabel('Values')

    axes[1].boxplot(metric_2, patch_artist=True)
    axes[1].set_title('False Positive Rate')

    axes[2].boxplot(metric_3, patch_artist=True)
    axes[2].set_title('True Positive Rate')

    # Ajustar el espacio entre subplots
    plt.tight_layout()

    plt.savefig(file_name+"_config_"+config+'.png')
    
    
def plot_roc_curves(roc_curves):
    plt.figure(figsize=(10, 8))
    for i, (fpr, tpr, roc_auc) in enumerate(roc_curves):
        plt.plot(fpr, tpr, label=f'Fold {i+1} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Across K-Folds')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.6, linestyle='--')
    plt.savefig("kfold_roc_curves.png")  # Opcional: guardar la figura
    plt.show()
