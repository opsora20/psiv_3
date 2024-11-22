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
from datasets import PatchClassifierDataset
from torch import device
import torch
from collections import defaultdict
import pandas as pd




def kfold_patient_classifier(model: PatchClassifier, dataset: PatchClassifierDataset, device: device,
                           df: pd.DataFrame, batch_size: int, k: int, show_fred=False,
                           show_roc=False):
    patches = dataset.images
    patches = torch.from_numpy(patches).float()
    labels = dataset.labels
    labels[labels == -1] = 0
    patients = dataset.patients

    sgkf = StratifiedGroupKFold(n_splits=k)
    best_thresholds_list = []
    train_metrics = []
    test_metrics = []
    
    for fold, (train_index, test_index) in enumerate(sgkf.split(patches, labels, patients)):
        
        print("Fold:",fold)
        patches_train = patches[train_index]
        patches_test = patches[test_index]
        true_labels_train = labels[train_index]
        true_labels_test = labels[test_index]
        patients_train = patients[train_index]
        patients_test = patients[test_index]

        print("Train_dim:",patches_train.shape[0])
        predicted_labels_train = compute_patches(model, device, patches_train, batch_size, test=True)
        predicted_proportion_train = patients_positive_proportion(patients_train, predicted_labels_train)
        
        df_train = df[df['CODI'].isin(predicted_proportion_train.keys())]
        df_train['PROPORTION'] = df_train['CODI'].map(predicted_proportion_train)
        
        densitats_train = list(df_train['DENSITAT'])
        proportions_train = list(df_train['PROPORTION'])

        print(type(densitats_train))
        print(type(proportions_train))
        print(densitats_train)
        print(proportions_train)
        best_threshold, train_fpr, train_tpr = compute_train_roc(proportions_train, densitats_train, fold, show_roc)

        predicted_labels_test = compute_patches(model, device, patches_test, batch_size, test=True)
        predicted_proportion_test = patients_positive_proportion(patients_test, predicted_labels_test)
        
        df_test = df[df['CODI'].isin(predicted_proportion_test.keys())]
        df_test['PROPORTION'] = df_test['CODI'].map(predicted_proportion_test)
        
        densitats_test = list(df_test['DENSITAT'])
        proportions_test = list(df_test['PROPORTION'])
        
        pred_densitat_test = (proportions_test > best_threshold).astype(int)
        
        test_acc, test_fpr, test_tpr = obtain_test_metrics(densitats_test, pred_densitat_test)
        
        train_metrics.append((best_threshold, train_fpr, train_tpr))
        test_metrics.append((test_acc, test_fpr, test_tpr))

    return train_metrics, test_metrics
        
        

def kfold_patch_classifier(model: PatchClassifier, dataset: PatchClassifierDataset, device: device,
                           batch_size: int, k: int, show_fred=False, show_roc=False):
    patches = dataset.images
    patches = torch.from_numpy(patches).float()
    labels = dataset.labels
    labels[labels == -1] = 0
    patients = dataset.patients

    sgkf = StratifiedGroupKFold(n_splits=k)
    train_metrics = []
    test_metrics = []
    
    for fold, (train_index, test_index) in enumerate(sgkf.split(patches, labels, patients)):
        
        print("Fold:",fold)
        patches_train = patches[train_index]
        patches_test = patches[test_index]
        labels_train = labels[train_index]
        labels_test = labels[test_index]
        patients_train = labels[train_index]
        patients_test = labels[test_index]
        
        print("Train_dim:",patches_train.shape[0])
        
        fred_list = compute_patches(model, device, patches_train, batch_size, train=True)
        best_threshold, train_fpr, train_tpr = compute_train_roc(fred_list, labels_train, fold, show_roc)
        print("Threshold",best_threshold, "FPR", train_fpr, "TPR", train_tpr)
        model.threshold = best_threshold
        predicted_labels = compute_patches(model, device, patches_test, batch_size, test=True)
        print("Test_dim:",patches_test.shape[0])
        test_acc, test_fpr, test_tpr = obtain_test_metrics(predicted_labels, labels_test)
        print("Accuracy",test_acc, "FPR", test_fpr, "TPR", test_tpr)
        
        train_metrics.append((best_threshold, train_fpr, train_tpr))
        test_metrics.append((test_acc, test_fpr, test_tpr))

    return train_metrics, test_metrics

    
def compute_patches(model: PatchClassifier, device: device, patches: np.ndarray, batch_size: int, train=False, test=False):
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
                show_fred=False,
        )
                fred_list.append(fred_result)
                
            if test:
                decided_class = model.execute(
                input_image,
                output_image
        )       
                predicted_label.append(decided_class)    
            
        pos += batch_size
        last += batch_size
        
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
        
    
def compute_train_roc(fred_list: list[float], target_labels: list[int], fold: int, show = False):
    fpr, tpr, thr = roc_curve(target_labels, fred_list)
    if show:
        plot_roc(fpr, tpr, fold)
    best_threshold, best_fpr, best_tpr = get_best_thr(fpr, tpr, thr)
    
    return best_threshold, best_fpr, best_tpr


def plot_roc(false_positive_rates: np.ndarray, true_positive_rates: np.ndarray, fold: int):
    roc_auc = auc(false_positive_rates, true_positive_rates)
    # Plot the ROC curve
    plt.figure()  
    plt.plot(false_positive_rates, true_positive_rates, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for H. Pylori Patch Classification')
    plt.legend()
    plt.savefig("Roc_curve_fold"+str(fold)+".png")


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


def mean_kfold(metrics: list[tuple[float]]) -> tuple[float, float, float]:
    metric_1, metric_2, metric_3 = zip(*metrics)
    
    k = len(metric_1)
    mean_1 = 0
    mean_2 = 0
    mean_3 = 0
    for m1, m2, m3 in zip(metric_1, metric_2, metric_3):
        mean_1 += m1
        mean_2 += m2
        mean_3 += m3
    mean_1 /= k
    mean_2 /= k
    mean_3 /= k
    
    return mean_1, mean_2, mean_3


def kfold_boxplot(metrics: list[tuple[float]], title_1: str, file_name: str):
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

    plt.savefig(file_name+'.png')


def patients_positive_proportion(patients: list[str], labels: list[int]):
    patients_labels = defaultdict(list)

    for patient, label in zip(patients, labels):
        patients_labels[patient].append(label)

    patients_labels = dict(patients_labels)
    
    positive_proportion = {}

    for patient, label_list in patients_labels.items():
        proportion = sum(label_list) / len(label_list) if len(label_list) > 0 else 0
        positive_proportion[patient] = proportion

    return positive_proportion