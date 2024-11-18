import numpy as np
import torch
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from math import sqrt, inf
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv
from torch.nn import MSELoss
import sys 

def test_autoencoder(model, batch_size, device, loader, threshold):
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
            
            fred_result = fred(input, output, plot = False)
            fred_list.append(fred_result)
            target_labels.append(label)
    roc(fred_list, target_labels, plot = True)
            
        
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
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        d = dist_thr(fpr, tpr)
        if (d < min_dist):
            best_thr = threshold
            min_dist = d
        threshold+=0.02
    if(plot):
        roc_plot(fpr_list, tpr_list)
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

    fred_result = num/den
    return fred_result

    