# -*- coding: utf-8 -*- noqa
"""
Created on Sat Nov 16 19:55:15 2024

@author: Joel tapia Salvador
"""
import os
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import numpy as np
import cv2
from cv2 import cvtColor, COLOR_RGB2HSV, COLOR_BGR2HSV, COLOR_RGB2BGR
import torch


def echo(out: str = "", *outs: str, **kwargs):
    """
    Print to console in realtime.

    Parameters
    ----------
    sep : string, optional
        String inserted between values. The default is " ".
    end : strings
        String appended after the last value. The default is newline.

    Raises
    ------
    TypeError
        Arguments badly given.

    Returns
    -------
    None.

    """
    out = str(out)

    try:
        outs = " ".join(outs)

        if outs != "":
            out = out + " " + outs

    except TypeError as error:
        raise TypeError("One or more of arguments is not a string.") from error

    os.system(f"echo '{out}'")


def generar_boxplot(data, titulo='Boxplot', nombre_archivo=None):
    """
    Genera un boxplot a partir de una lista de datos.

    Parámetros:
        data (list): Lista de datos para generar el boxplot.
        titulo (str): Título del boxplot. Por defecto es 'Boxplot'.
        nombre_archivo (str): Nombre del archivo para guardar la imagen (opcional). 
                              Si es None, no se guarda la gráfica.
    """
    plt.figure(figsize=(8, 6))  # Tamaño del gráfico
    plt.boxplot(data, patch_artist=True, boxprops=dict(
        facecolor='lightblue'))  # Boxplot con estilo
    plt.title(titulo)
    plt.ylabel('Valores')
    plt.grid(True, linestyle='--', alpha=0.7)  # Opcional: Añadir cuadrícula

    # Mostrar la gráfica
    plt.show()

    # Guardar la gráfica si se proporciona un nombre de archivo
    if nombre_archivo:
        plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
        print(f"Boxplot guardado como {nombre_archivo}")


def load_patient_diagnosis(path_patient_diagnosis: str) -> pd.DataFrame:
    csv_patient_diagnosis = pd.read_csv(path_patient_diagnosis)
    csv_patient_diagnosis["DENSITAT"][csv_patient_diagnosis["DENSITAT"] == "ALTA"] = 1
    csv_patient_diagnosis["DENSITAT"][csv_patient_diagnosis["DENSITAT"] == "BAIXA"] = 1
    csv_patient_diagnosis["DENSITAT"][csv_patient_diagnosis["DENSITAT"] == "NEGATIVA"] = 0

    return csv_patient_diagnosis


def save_pickle(object, path: str):
    # try:
    #     with open(path, "rb") as archivo:
           
    #         print("El archivo ya existe")

    # except FileNotFoundError:
    with open(path, "wb") as archivo:
        pickle.dump(object, archivo)
    print("Archivo nuevo creado.")
        
def load_pickle(path: str):
    """
    Carga los datos de un archivo pickle. 
    Si el archivo no existe, devuelve un diccionario vacío.

    :param nombre_archivo: Nombre del archivo pickle a cargar.
    :return: Diccionario cargado o un diccionario vacío si no existe.
    """
    try:
        with open(path, "rb") as archivo:
            datos = pickle.load(archivo)
            print(f"Datos cargados desde '{path}'.")
            return datos
    except FileNotFoundError:
        print(f"El archivo '{path}' no existe.")
        return
    except pickle.UnpicklingError:
        print(f"El archivo '{path}' no se puede deserializar correctamente."
              + " Verifica su contenido.")
        return


def compare_histograms(avg_input_histogram, avg_output_histogram, bin_edges):

    plt.figure(figsize=(10, 6))
    plt.plot(bin_edges[:-1], avg_input_histogram, label="Input Batch Histogram", color="blue", linestyle="--")
    plt.plot(bin_edges[:-1], avg_output_histogram, label="Output Batch Histogram", color="orange", linestyle="-")
    plt.title("Hue Histogram Comparison for Batch")
    plt.xlabel("Hue Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()


def get_all_embeddings(patient_dict: dict, dataset, encoder, output_size, device, loader, max_images, is_resnet = False):
    to_remove = []
    encoder.eval()

    for patient in patient_dict.keys():
        process = dataset.load_patient(patient, max_images)
        if process:
            # Inicializa un tensor vacío para ir acumulando los resultados
            patient_patches = None

            for idx, patches in enumerate(loader["train"]):
                patches = patches.to(device)
                with torch.no_grad():  # Evita almacenar gradientes
                    if(is_resnet == False):
                        embeddings = encoder.get_embeddings(patches, output_size)
                    else:
                        embeddings = encoder(patches)

                # Si es la primera iteración, inicializa patient_patches
                if patient_patches is None:
                    patient_patches = embeddings.cpu()
                else:
                    # Concatenar el nuevo batch al tensor acumulado
                    patient_patches = torch.cat((patient_patches, embeddings.cpu()), dim=0)


            # Guarda el tensor acumulado en el diccionario
            patient_dict[patient]['patches'] = patient_patches
            print(patient, patient_dict[patient]['patches'].shape)
        else:
            to_remove.append(patient)

    # Eliminar pacientes no procesados
    for patient in to_remove:
        del patient_dict[patient]
    return patient_dict

