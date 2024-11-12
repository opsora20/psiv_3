import os
import pandas as pd
import io
import numpy as np
from torch.utils.data import Dataset, DataLoader


def load_cropped_patients(cropped_dir, metadata_path):
   
    # Cargar los datos desde los archivos CSV
    cropped_csv = pd.read_csv(metadata_path)
    
    imgs = []

    
    for patient_dir in os.listdir(cropped_dir):
        aux = patient_dir[:-2]
        dens = cropped_csv[cropped_csv["CODI"] == aux]["DENSITAT"]
        if(dens == "NEGATIVA"):
            for file_img in os.listdir(os.join(cropped_dir, patient_dir)):
                image = io.imread(file_img)
                imgs.append(image)
    
    return np.array(imgs)


def create_dataloaders(class_dataset, batch):
    return DataLoader(class_dataset, batch_size = batch, shuffle = True, num_workers = 4)
