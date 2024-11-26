# -*- coding: utf-8 -*- noqa
"""
Created on Tue Nov 12 12:44:33 2024

@author: joanb
"""

import os

import numpy as np
import pandas as pd

from skimage import io, color
from utils import echo


def load_cropped_patients(path_root_directory: str, info: pd.DataFrame):
    """
    Load Cropped Dataset.

    Parameters
    ----------
    path_root_directory : str
        DESCRIPTION.
    info : pd.DataFrame
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    imgs = []

    for patient_directory in os.listdir(path_root_directory):
        aux = patient_directory[:-2]
        dens = info[info["CODI"] == aux]["DENSITAT"].iloc[0]
        if (dens == "NEGATIVA"):
            echo(f'Reading: {patient_directory}')
            for image_file in os.listdir(os.path.join(
                    path_root_directory,
                    patient_directory
            )):
                if (image_file.endswith(".png")):
                    image = io.imread(os.path.join(
                        path_root_directory, patient_directory, image_file))
                    image = color.rgba2rgb(image)
                    # echo(image.shape)
                    if (image.shape[0] != 256 or image.shape[1] != 256):
                        echo(f'- {image_file}: {image.shape}')
                    else:
                        # echo(f'+ {file_img}')
                        image = image.transpose(2, 0, 1)
                        imgs.append(image)

    return np.array(imgs)


def load_patient_images(patient, root_dir, maximages = 1000):
    patient_dir = os.path.join(root_dir, patient)
    patches = []
    for i in range(2):
        patient_dir_full = patient_dir+"_"+str(i)
        if(os.path.exists(patient_dir_full)):
            echo(f'Reading: {patient+"_"+str(i)}')
            for image_file in os.listdir(patient_dir_full):
                if (image_file.endswith(".png")):
                    image = io.imread(os.path.join(
                        patient_dir_full, image_file))
                    image = color.rgba2rgb(image)
                    # echo(image.shape)
                    if (image.shape[0] != 256 or image.shape[1] != 256):
                        echo(f'- {image_file}: {image.shape}')
                    else:
                        # echo(f'+ {file_img}')
                        image = image.transpose(2, 0, 1)
                        patches.append(image)
                if(len(patches) >= maximages):
                    break
    return np.array(patches)


def load_annotated_patients(path_root_directory, info: pd.DataFrame):
    """
    Load Annotated Dataset.

    Parameters
    ----------
    path_root_directory : TYPE
        DESCRIPTION.
    info : pd.DataFrame
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    patches = []
    patient_list = []
    labels_list = []
    for idx, row in info.iterrows():
        patient_directory = row["Pat_ID"] + "_" + str(row["Section_ID"])
        window = row["Window_ID"]
        if (isinstance(window, int)):
            window = ("0000" + str(window))[-5:]
        else:
            window = ("0000" + window)[-10:]
        file_path = os.path.join(
            path_root_directory, patient_directory, window + ".png")
        if row["Presence"] != 0:
            if os.path.isfile(file_path):
                image = io.imread(file_path)
                if image.ndim == 2:
                    image = np.stack((image,)*3, axis=-1)
                elif image.shape[2] > 3:
                    image = color.rgba2rgb(image)
                if image.shape[0] != 256 or image.shape[1] != 256:
                    echo(f"Tamaño incorrecto para {file_path}: {image.shape}")
                    continue
                else:
                    image = image.transpose(2, 0, 1)
                    patches.append(image)
                    patient_list.append(row["Pat_ID"])
                    labels_list.append(row["Presence"])

        """"Imagenes no encontradas en el directorio"""
        # else:
        #     echo(f"Archivo no encontrado: {file_path}")

    return np.array(patches), np.array(patient_list), np.array(labels_list)
