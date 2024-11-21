# -*- coding: utf-8 -*- noqa
"""
Created on Tue Nov 12 12:44:33 2024

@author: joanb
"""

import numpy as np
import torch

import matplotlib.pyplot as plt

from copy import deepcopy
from skimage.color import rgb2hsv


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
            show: bool = False,
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

        input_hsv = rgb2hsv(input_image)
        output_hsv = rgb2hsv(output_image)

        input_hue = input_hsv[:, :, 0]
        output_hue = output_hsv[:, :, 0]

        num = np.sum((input_hue >= 0.95) | (input_hue <= 0.05))
        den = np.sum((output_hue >= 0.95) | (output_hue <= 0.05))

        selected_hue = deepcopy(output_hue)

        selected_hue[(selected_hue >= 0.95) | (selected_hue <= 0.05)]

        if show:
            plt.figure(figsize=(10, 5))

            # Primer subplot
            plt.subplot(1, 3, 1)  # (n_filas, n_columnas, índice)
            plt.imshow(input_hue)
            plt.title('Imagen 1')
            plt.axis('off')  # Ocultar ejes

            # Segundo subplot
            plt.subplot(1, 3, 2)
            plt.imshow(output_hue)
            plt.title('Imagen 2')
            plt.axis('off')

            # Tercer subplot
            plt.subplot(1, 3, 3)
            plt.imshow(selected_hue)
            plt.title('Imagen 3')
            plt.axis('off')

            # Mostrar la figura
            plt.tight_layout()  # Ajusta el espacio entre subplots
            plt.show()

        fred = num / den

        return fred

    def classify(self, fred):
        if fred > self.__threshold:
            return 1
        return 0

    def encode(self, input_image):
        input_image.to(self.__device)
        return self.__autoencoder(input_image)

    def execute(self, input_image):
        output_image = self.encode(input_image)

        fred = self.calculate_fred(input_image, output_image, False)

        decided_class = self.classify(fred)

        return decided_class
