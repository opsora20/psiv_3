import os
import pandas as pd
from skimage import io, color
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils import echo


def load_cropped_patients(cropped_dir, cropped_csv):

    imgs = []

    for patient_dir in os.listdir(cropped_dir):
        aux = patient_dir[:-2]
        dens = cropped_csv[cropped_csv["CODI"] == aux]["DENSITAT"].iloc[0]
        if (dens == "NEGATIVA"):
            echo(f'Reading: {patient_dir}')
            for file_img in os.listdir(os.path.join(cropped_dir, patient_dir)):
                if (file_img.endswith(".png")):
                    image = io.imread(os.path.join(
                        cropped_dir, patient_dir, file_img))
                    image = color.rgba2rgb(image)
                    # echo(image.shape)
                    if (image.shape[0] != 256 or image.shape[1] != 256):
                        echo(f'- {file_img}: {image.shape}')
                    else:
                        # echo(f'+ {file_img}')
                        image = image.transpose(2, 0, 1)
                        imgs.append(image)

    return np.array(imgs)


def load_annotated_patients(annotated_dir, annotated_excel):
    patches = []
    patient_list = []
    labels_list = []
    for idx, row in annotated_excel.iterrows():
        patient_dir = row["Pat_ID"] + "_" + str(row["Section_ID"])
        window = row["Window_ID"]
        if (isinstance(window, int)):
            window = ("0000" + str(window))[-5:]
        else:
            window = ("0000" + window)[-10:]
        file_path = os.path.join(annotated_dir, patient_dir, window+".png")
        if row["Presence"] == -1:
            if os.path.isfile(file_path):
                im = io.imread(file_path)
                if im.ndim == 2:
                    im = np.stack((im,)*3, axis=-1)
                elif im.shape[2] > 3:
                    im = color.rgba2rgb(im)
                if im.shape[0] != 256 or im.shape[1] != 256:
                    echo(f"Tamaño incorrecto para {file_path}: {im.shape}")
                    continue
                else:
                    im = im.transpose(2, 0, 1)
                    patches.append(im)
                    patient_list.append(row["Pat_ID"])
                    labels_list.append(row["Presence"])

        """"Imagenes no encontradas en el directorio"""
        # else:
        #     echo(f"Archivo no encontrado: {file_path}")

    return np.array(patches), np.array(patient_list), np.array(labels_list)


def create_dataloaders(class_dataset, batch):
    return DataLoader(class_dataset, batch_size=batch, shuffle=True)
