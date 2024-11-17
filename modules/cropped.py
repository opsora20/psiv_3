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
                        echo(file_img)
                        echo(image.shape)
                    else:
                        image = image.transpose(2, 0, 1)
                        imgs.append(image)

    return np.array(imgs)


def load_annotated_patients(annotated_dir, annotated_excel):
    patches = []
    labels_list = []
    df_annotations = pd.read_excel(annotated_excel)
    for idx, row in df_annotations.iterrows():
        file = row['FILENAME']
        label = row['LABEL']
        file_path = os.path.join(annotated_dir, file)
        if os.path.isfile(file_path):
            im = io.imread(file_path)
            if im.ndim == 2:
                im = np.stack((im,)*3, axis=-1)
            elif im.shape[2] > 3:
                im = im[:, :, :3]
            if im.shape[0] != 256 or im.shape[1] != 256:
                echo(f"Tamaño incorrecto para {file}: {im.shape}")
                continue
            else:
                im = im.transpose(2, 0, 1)
                patches.append(im)
                labels_list.append(label)
        else:
            echo(f"Archivo no encontrado: {file_path}")

    return np.array(patches), np.array(labels_list)


def create_dataloaders(class_dataset, batch):
    return DataLoader(class_dataset, batch_size=batch, shuffle=True)
