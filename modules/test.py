import numpy as np
import torch
#from sklearn.metrics import roc_curve, auc
#from sklearn.preprocessing import label_binarize
from math import sqrt, inf
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv
from torch.nn import MSELoss

def test_autoencoder(model, batch_size, device, loader, threshold):
    model.eval()
    target_labels = []
    pred_labels = []
    fred_list = []
    for batch_id, inputs in enumerate(loader["val"]):
        if batch_id == 1:
            break
        if batch_id%100 == 0:
            print (batch_id)
        inputs = inputs.to(device)
        outputs = model(inputs)
        loss = MSELoss()
        l = loss(inputs, outputs)
        print(l)
        for input, output in zip(inputs, outputs):
            input = np.transpose(input.cpu().detach().numpy(), axes=(1, 2, 0))
            output = np.transpose(output.cpu().detach().numpy(), axes=(1, 2, 0))
            plt.figure(figsize=(10, 5))

            # Primer subplot
            plt.subplot(1, 2, 1)  # (n_filas, n_columnas, índice)
            plt.imshow(input)
            plt.title('Imagen 1')
            plt.axis('off')  # Ocultar ejes

            # Segundo subplot
            plt.subplot(1, 2, 2)
            plt.imshow(output)
            plt.title('Imagen 2')
            plt.axis('off')

            # Mostrar la figura
            plt.tight_layout()  # Ajusta el espacio entre subplots
            plt.show()


            # fred_result = fred(input, output)
            # print(fred_result)
            # fred_list.append(fred_result)
            # if fred > threshold:
            #     pred_labels.append[1]
            # else:
            #     pred_labels.append[0]
            
        
            
    # # Crear el histograma
    # plt.hist(fred_list, bins=30, color='skyblue', edgecolor='black')

    # # Agregar etiquetas y título
    # plt.xlabel('Valor')
    # plt.ylabel('Frecuencia')
    # plt.title('Histograma de la lista de números')

    # # Mostrar el gráfico
    # plt.savefig('histograma.png')
    # plt.close()
            

                
    #         target_labels.append[label]
    # target_labels = np.array(target_labels)
    # target_labels[target_labels == -1] = 0
    # pred_labels = np.array(pred_labels)

        
def roc(target_labels, pred_labels, plot=False):
    fpr_arr, tpr_arr, thr_arr =  roc_curve(target_labels, pred_labels)
    dis_min = inf
    best_thr = None
    best_point = (None, None)
    for fpr, tpr, thr in zip(fpr_arr, tpr_arr, thr_arr):
        dis = sqrt(fpr**2 + (tpr-1) ** 2)
        if dis_min < dis:
            dis_min = dis
            best_thr = thr
            best_point = (fpr, tpr)
    
    if plot:
        roc_plot(fpr_arr, tpr_arr)
            

    
    return best_point, best_thr
        
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
    
    


def fred(input: torch.Tensor, output: torch.Tensor) -> float: 
    input = np.transpose(input.cpu().detach().numpy(), axes=(1, 2, 0))
    output = np.transpose(output.cpu().detach().numpy(), axes=(1, 2, 0))

    
    input_hsv = rgb2hsv(input)
    output_hsv = rgb2hsv(output)
    
    input_hue = input_hsv[:,:,0]
    output_hue = output_hsv[:,:,0]
    
    num = np.sum((input_hue >= -20) & (input_hue <= 20))
    den = np.sum((output_hue >= -20) & (output_hue <= 20))
    
    fred_result = num/den
    return fred_result

    