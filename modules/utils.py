# -*- coding: utf-8 -*- noqa
"""
Created on Sat Nov 16 19:55:15 2024

@author: Joel tapia Salvador
"""
import os
import matplotlib.pyplot as plt
import pandas as pd
import pickle

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

    try:
        # Intentar abrir y cargar el archivo
        with open(path, "rb") as archivo:
            diccionario_cargado = pickle.load(archivo)
        print("Diccionario cargado:", diccionario_cargado)

    except FileNotFoundError:
        # Si el archivo no existe, inicializar un diccionario vacío y guardarlo
        print(f"El archivo '{path}' no existe. Creando uno nuevo...")
        diccionario_cargado = {}
        with open(path, "wb") as archivo:
            pickle.dump(diccionario_cargado, archivo)
        print("Archivo nuevo creado con un diccionario vacío.")
        
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
        print(f"El archivo '{path}' no se puede deserializar correctamente. Verifica su contenido.")
        return