import os
import pandas as pd
from skimage import io
import numpy as np
from torch.utils.data import Dataset, DataLoader


def load_cropped_patients(cropped_dir, cropped_csv):
    
    imgs = []

    
    for patient_dir in os.listdir(cropped_dir):
        aux = patient_dir[:-2]
        dens = cropped_csv[cropped_csv["CODI"] == aux]["DENSITAT"].iloc[0]
        if(dens == "NEGATIVA"):
            for file_img in os.listdir(os.path.join(cropped_dir, patient_dir)):
                if(file_img.endswith(".png")):
                    image = io.imread(os.path.join(cropped_dir, patient_dir, file_img))
                    imgs.append(image)
        
    return np.array(imgs)


def create_dataloaders(class_dataset, batch):
    return DataLoader(class_dataset, batch_size = batch, shuffle = True)
