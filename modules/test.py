import numpy as np
import torch
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from math import sqrt, inf
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv
from torch.nn import MSELoss
import sys 
from sklearn.model_selection import StratifiedGroupKFold
import pickle

def calculations(model, device, loader):
    model.eval()
    target_labels = []
    pred_labels = []
    fred_list = []
    for batch_id, (inputs, labels) in enumerate(loader["val"]):
        if batch_id%100 == 0:
            print (batch_id)
        inputs = inputs.to(device)
        outputs = model(inputs)
        for input, output, label in zip(inputs, outputs, labels):
            #print(label)
            fred_result = fred(input, output, plot = False)
            fred_list.append(fred_result)
            target_labels.append(label)
    return target_labels, pred_labels, fred_list

def test_autoencoder(model, device, loader, load = False):
    if load:
        with open("data.pk", "rb") as file:
            data = pickle.load(file)
            target_labels = data["target_labels"]
            target_labels = np.array(target_labels)
            target_labels[target_labels==-1] = 0
            print(target_labels)
            pred_labels = data["pred_labels"]
            fred_list = data["fred_list"]
            print(fred_list)
    else:
        target_labels, pred_labels, fred_list = calculations(model, device, loader)
        with open("data.pk", "wb") as file:
            data = {
                "target_labels": target_labels,
                "pred_labels": pred_labels,
                "fred_list": fred_list,
            }
            
            pickle.dump(data, file)
        
        
    fpr, tpr, thr = roc_curve(target_labels, fred_list)
    roc_auc = auc(fpr, tpr)
    # Plot the ROC curve
    plt.figure()  
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Breast Cancer Classification')
    plt.legend()
    plt.show()
    # roc(fred_list, target_labels, plot = True)
            
        
    """
    # # Crear el histograma
    plt.hist(fred_1_list, bins=30, color='skyblue', edgecolor='black')

    # # Agregar etiquetas y título
    plt.xlabel('Valor')
    plt.ylabel('Frecuencia')
    plt.title('Histograma de la lista de números')

    # # Mostrar el gráfico
    plt.savefig('histograma_positive.png')
    plt.close()

    # # Crear el histograma
    plt.hist(fred_2_list, bins=30, color='skyblue', edgecolor='black')

    # # Agregar etiquetas y título
    plt.xlabel('Valor')
    plt.ylabel('Frecuencia')
    plt.title('Histograma de la lista de números')

    # # Mostrar el gráfico
    plt.savefig('histograma_negative.png')
    plt.close()
    """
    
def patient_kfold(model, device, batch_size, patches, labels, patients, k, config):
    patches = torch.from_numpy(patches).float()
    sgkf = StratifiedGroupKFold(n_splits=k)
    for fold, (train_index, test_index) in enumerate(sgkf.split(patches, labels, patients)):
        fred_list = []
        labels_list = []
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
            outputs_batch = model(inputs_batch)
            
            for inp, out, label in zip(inputs_batch, outputs_batch, labels_batch):
                fred_result = fred(inp, out, plot = False)
                fred_list.append(fred_result)
                if(label == -1):
                    labels_list.append(0)
                elif(label == 1):
                    labels_list.append(label)
                else:
                    labels_list.append(0)
                

            pos += batch_size
            last += batch_size
        #th = roc(fred_list, labels, plot = True)
        fpr, tpr, thr = roc_curve(labels_list, fred_list)
        roc_auc = auc(fpr, tpr)
        # Plot the ROC curve
        plt.figure()  
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Breast Cancer Classification')
        plt.legend()
        plt.savefig('Roc_curve_config'+config+'_fold'+str(fold)+'.png')
        #print("Best_threshold:", thr)
        
        
def roc(freds, target_labels, plot=False):
    threshold = 0
    tpr_list = []
    fpr_list = []
    min_dist = sys.maxsize
    while threshold < 1:
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for fr, tl in zip(freds, target_labels):
            if(fr >= threshold):
                if(tl == 1):
                    tp+=1
                else:
                    fp+=1
            else:
                if(tl == -1):
                    tn += 1
                else:
                    fn +=1
        if tp == 0:
            tpr = 0
        else: 
            tpr = tp / (tp + fn)
        if fp == 0:
            fpr = 0
        else:
            fpr = fp / (fp + tn)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        d = dist_thr(fpr, tpr)
        if (d < min_dist):
            best_thr = threshold
            min_dist = d
        threshold+=0.02
    return best_thr

def dist_thr(fpr, tpr):
    return pow(pow(fpr, 2) + pow(1-tpr, 2), 0.5)

def roc_plot(fpr_arr, tpr_arr):
    roc_auc = auc(fpr_arr, tpr_arr)
    plt.figure()
    plt.plot(fpr_arr, tpr_arr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Línea de referencia (aleatoria)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.title('Curva ROC')
    plt.legend(loc='lower right')
    plt.show()

def fred(input: torch.Tensor, output: torch.Tensor, plot = False) -> float: 
    input = np.transpose(input.cpu().detach().numpy(), axes=(1, 2, 0))
    output = np.transpose(output.cpu().detach().numpy(), axes=(1, 2, 0))

    input_hsv = rgb2hsv(input)
    output_hsv = rgb2hsv(output)
    

    if plot:
        plt.figure(figsize=(10, 5))

        # Primer subplot
        plt.subplot(1, 2, 1)  # (n_filas, n_columnas, índice)
        plt.imshow(input_hsv[:,:,0])
        plt.title('Imagen 1')
        plt.axis('off')  # Ocultar ejes

        # Segundo subplot
        plt.subplot(1, 2, 2)
        plt.imshow(output_hsv[:,:,0])
        plt.title('Imagen 2')
        plt.axis('off')

        # Mostrar la figura
        plt.tight_layout()  # Ajusta el espacio entre subplots
        plt.show()


    input_hue = input_hsv[:,:,0]
    output_hue = output_hsv[:,:,0]
    
    num = np.sum((input_hue >= 0.95) | (input_hue <= 0.05))
    den = np.sum((output_hue >= 0.95) | (output_hue <= 0.05))
    #print(num, den)

    if den == 0:
    #     #print("NO HAY ROJO EN LA IMAGEN DE SALIDA")
        fred_result = 0
    else:
        fred_result = num/den
    #     #print(fred_result)
    return fred_result



        
    
    