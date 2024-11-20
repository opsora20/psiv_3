# -*- coding: utf-8 -*- noqa
"""
Created on Sat Nov 16 19:55:15 2024

@author: Joel tapia Salvador
"""
import os
import matplotlib.pyplot as plt

def echo(out):
    os.system(f"echo '{out}'")

import matplotlib.pyplot as plt

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
    plt.boxplot(data, patch_artist=True, boxprops=dict(facecolor='lightblue'))  # Boxplot con estilo
    plt.title(titulo)
    plt.ylabel('Valores')
    plt.grid(True, linestyle='--', alpha=0.7)  # Opcional: Añadir cuadrícula
    
    # Mostrar la gráfica
    plt.show()
    
    # Guardar la gráfica si se proporciona un nombre de archivo
    if nombre_archivo:
        plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
        print(f"Boxplot guardado como {nombre_archivo}")
