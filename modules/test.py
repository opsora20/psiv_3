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



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16
k_folds = 5

output_dir = "./kfold_results"
os.makedirs(output_dir, exist_ok=True)

patient_ids = data._data['Pat_ID'].values

gkf = GroupKFold(n_splits=k_folds)
fold = 1

best_thresholds = []

for train_index, val_index in gkf.split(data, groups=patient_ids):
    print(f'Fold {fold}/{k_folds}')
    
    val_subset = torch.utils.data.Subset(data, val_index)
    
    val_loader = create_dataloaders(val_subset, batch_size)
    
    config = '1'
    net_paramsEnc, net_paramsDec, inputmodule_paramsDec, inputmodule_paramsEnc = AEConfigs(config)
    model = AutoEncoderCNN(inputmodule_paramsEnc, net_paramsEnc, inputmodule_paramsDec, net_paramsDec)
    model.load_state_dict(torch.load(os.path.join(output_dir, f"autoencoder_fold_{fold}.pth"), map_location=device))
    model.to(device)
    model.eval()
    
    fred_list = []
    target_labels = []
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        for input, output, label in zip(inputs, outputs, labels):
            fred_result = fred(input, output, plot=False)
            fred_list.append(fred_result)
            target_labels.append(label)
    
    fpr, tpr, thresholds = roc_curve(target_labels, fred_list)
    optimal_idx = np.argmax(tpr - fpr)
    best_threshold = thresholds[optimal_idx]
    best_thresholds.append(best_threshold)
    print(f'Mejor umbral para el fold {fold}: {best_threshold}')
    
    predictions = [1 if x >= best_threshold else 0 for x in fred_list]
    correct = sum([1 if pred == true else 0 for pred, true in zip(predictions, target_labels)])
    accuracy = correct / len(target_labels)
    print(f'Exactitud para el fold {fold}: {accuracy}')
    
    with open(os.path.join(output_dir, f"results_fold_{fold}.txt"), "w") as f:
        f.write(f"Mejor umbral para el fold {fold}: {best_threshold}\n")
        f.write(f"Exactitud para el fold {fold}: {accuracy}\n")
    
    gc.collect()
    torch.cuda.empty_cache()

    fold += 1

print("Evaluación completada para todos los folds.")

    
