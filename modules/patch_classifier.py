# -*- coding: utf-8 -*- noqa
"""
Created on Tue Nov 12 12:44:33 2024

@author: joanb
"""

import numpy as np
import torch
import sys

import matplotlib.pyplot as plt

from copy import deepcopy
import cv2
from cv2 import cvtColor, COLOR_RGB2HSV, COLOR_BGR2HSV, COLOR_RGB2BGR


class PatchClassifier():
    def __init__(self, autoencoder, device, threshold=None):
        self.__autoencoder = autoencoder
        self.__device = device
        self.__threshold = threshold

        self.__autoencoder.to(device)

        # Congelar autoencoder
        self.__autoencoder.eval()

    def calculate_fred(
            self,
            input_image: torch.Tensor,
            output_image: torch.Tensor,
            show_fred: bool = False,
    ) -> float:
        """
        Calculate the fred of the images.

        Parameters
        ----------
        input_image : torch tensor
            DESCRIPTION.
        output_image : torch tensor
            DESCRIPTION.
        show : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        float
            DESCRIPTION.

        """
        input_image = np.transpose(
            input_image.cpu().detach().numpy(), axes=(1, 2, 0))
        output_image = np.transpose(
            output_image.cpu().detach().numpy(), axes=(1, 2, 0))


        input_hsv = cvtColor(input_image, COLOR_RGB2HSV)
        output_hsv = cvtColor(output_image, COLOR_RGB2HSV)

        # print(input_image)
        # print(output_image)
        
        input_hue = input_hsv[:, :, 0]
        output_hue = output_hsv[:, :, 0]
                
        
        # top = (input_hue <= 70) | (input_hue >= 350)
        # bot = (output_hue <= 70) | (output_hue >= 350)
        # mascara_input = top.astype(int)        
        # mascara_input = np.expand_dims(mascara_input, axis=-1)
        # mascara_input = np.repeat(mascara_input, 3, axis=-1)
        
        # mascara_output = bot.astype(int)
        # mascara_output = np.expand_dims(mascara_output, axis=-1)
        # mascara_output = np.repeat(mascara_output, 3, axis=-1)

        
        num = np.sum((input_hue <= 40) | (input_hue >= 320))
        den = np.sum((output_hue <= 40) | (output_hue >= 320)) + 1
        
        

        if show_fred:
            plt.figure(figsize=(20, 10))

            # Primer subplot
            plt.subplot(1, 2, 1)  # (n_filas, n_columnas, índice)
            plt.imshow(input_image)
            plt.title('Imagen 1')
            plt.axis('off')  # Ocultar ejes

            # Segundo subplot
            plt.subplot(1, 2, 2)
            plt.imshow(output_image)
            plt.title('Imagen 2')
            plt.axis('off')
            
            # # Primer subplot
            # plt.subplot(2, 2, 3)  # (n_filas, n_columnas, índice)
            # plt.imshow(top_int, cmap='gray')
            # plt.title('Imagen 1')
            # plt.axis('off')  # Ocultar ejes

            # # Segundo subplot
            # plt.subplot(2, 2, 4)
            # plt.imshow(bot_int, cmap='gray')
            # plt.title('Imagen 2')
            # plt.axis('off')

            # Mostrar la figura
            plt.tight_layout()  # Ajusta el espacio entre subplots
            plt.show()


        fred = num / den
        
        # hist_input, bin_edges = np.histogram(input_hue, bins=50, range=(0,360), density=True)
        # hist_output, _ = np.histogram(output_hue, bins=50, range=(0,360), density=True)
        # plt.figure(figsize=(10, 6))
        # plt.plot(bin_edges[:-1], hist_input, label="Input Batch Histogram", color="blue", linestyle="--")
        # plt.plot(bin_edges[:-1], hist_output, label="Output Batch Histogram", color="orange", linestyle="-")
        # plt.title("Hue Histogram Comparison for Batch")
        # plt.xlabel("Hue Value")
        # plt.ylabel("Frequency")
        # plt.legend()
        # plt.show()

        return fred#, hist_input, hist_output, bin_edges

    def classify(self, fred):
        if fred > self.__threshold:
            return 1
        return 0

    def encode(self, input_image):
        input_image.to(self.__device)
        return self.__autoencoder(input_image)

    def execute(self, input_image, output_image):
        fred = self.calculate_fred(input_image, output_image, False)

        decided_class = self.classify(fred)

        return decided_class
    
    @property
    def threshold(self):
        """Getter para el atributo threshold."""
        return self.__threshold    
    
    @threshold.setter
    def threshold(self, new_threshold: float):
        """Setter para el atributo threshold."""
        if not isinstance(new_threshold, float):
            raise ValueError("El threshold debe de ser un float.")
        self.__threshold = new_threshold
