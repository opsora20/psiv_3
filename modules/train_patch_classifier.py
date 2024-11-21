# -*- coding: utf-8 -*- noqa
"""
Created on Wed Nov 20 23:06:19 2024

@author: JoelT
"""

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import sys
from patch_classifier import PatchClassifier
import torch



def train_patch_classifier(model: PatchClassifier, dataloader, device, batch_size, k, show_fred=False, show_roc=False):
    patches = dataloader.images
    patches = torch.from_numpy(patches).float()
    labels = dataloader.labels
    patients = dataloader.patients

    sgkf = StratifiedGroupKFold(n_splits=k)
    best_thresholds_list = []
    best_fpr_list = []
    best_tpr_list = []
    
    for fold, (train_index, test_index) in enumerate(sgkf.split(patches, labels, patients)):
        fred_list = []
        target_labels = []
        
        patches_train = patches[train_index]
        patches_test = patches[test_index]
        labels_train = labels[train_index]
        labels_test = labels[test_index]
        
        size_train = patches_train.shape[0]
        size_test = patches_test.shape[0]
        
        pos = 0
        last = batch_size
        
        while pos < size_train:
            if last > size_train:
                last = size_train
            labels_batch = labels_train[pos : last]
            inputs_batch = patches_train[pos : last]
            inputs_batch = inputs_batch.to(device)
            outputs_batch = model.encode(inputs_batch)
            
            for input_image, output_image, label in zip(inputs_batch, outputs_batch, labels_batch):
                fred_result = model.calculate_fred(
                input_image,
                output_image,
                show_fred=False,
            )
                fred_list.append(fred_result)
                if(label == -1):
                    target_labels.append(0)
                elif(label == 1):
                    target_labels.append(label)
                else:
                    target_labels.append(0)
                
            pos += batch_size
            last += batch_size
            
        best_threshold, best_fpr, best_tpr = compute_roc(fred_list, target_labels, show_roc)
        best_thresholds_list.append(best_threshold)
        best_fpr_list.append(best_fpr)
        best_tpr_list.append(best_tpr)
        
    mean_thr, mean_fpr, mean_tpr = mean_kfold(best_thresholds_list, best_fpr_list, best_tpr_list)

    return mean_thr, mean_fpr, mean_tpr
    
    
def compute_roc(fred_list: list[float], target_labels: list[int], show: bool):
    fpr, tpr, thr = roc_curve(target_labels, fred_list)
    if show:
        plot_roc(fpr, tpr)
    best_threshold, best_fpr, best_tpr = get_best_thr(fpr, tpr, thr)
    
    return best_threshold, best_fpr, best_tpr


def plot_roc(false_positive_rates: np.array, true_positive_rates: np.array):
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
    # plt.savefig('Roc_curve.png')


def get_best_thr(false_positive_rates: np.array, true_positive_rates: np.array, thresholds: np.array) -> tuple[float, float, float]:
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


def mean_kfold(thr_list: list[float], fpr_list: list[float], tpr_list: list[float]) -> tuple[float, float, float]:
    k = len(thr_list)
    mean_thr = 0
    mean_fpr = 0
    mean_tpr = 0
    for thr, fpr, tpr in zip(thr_list, fpr_list, tpr_list):
        mean_thr += thr
        mean_fpr += fpr
        mean_tpr += tpr
    mean_thr /= k
    mean_fpr /= k
    mean_tpr /= k
    
    return mean_thr, mean_fpr, mean_tpr
