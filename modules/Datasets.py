# -*- coding: utf-8 -*- noqa
"""
Created on Tue Nov 12 12:44:33 2024

@author: joanb
"""

import torch
import pickle

import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader

from load_datasets import load_cropped_patients, load_annotated_patients


class AutoEncoderDataset(Dataset):

    def __init__(
            self,
            path_info_file: str,
            dataset_root_directory: str,
            transform=None,
            pickle_save_file: str = "",
            pickle_load_file: str = "",
    ):
        self.__info = pd.read_csv(path_info_file)
        self.__dataset_root_directory = dataset_root_directory
        self.__transform = transform

        if pickle_load_file != "":
            with open(pickle_load_file, "rb") as file:
                data = pickle.load(file)

                self.__images = data["__images"]

        else:
            self.__read_images()
            if pickle_save_file != "":
                with open(pickle_save_file, "wb") as file:
                    data = {"__images": self.__images}

                    pickle.dump(data, file)

    def __read_images(self):
        self.__images = load_cropped_patients(
            self.__dataset_root_directory, self.__info)

    def __len__(self):
        """
        Get amount of images loaded.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return len(self.__images)

    def __getitem__(self, idx):
        """
        Get item method.

        Parameters
        ----------
        idx : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.
        label_sample : TYPE
            DESCRIPTION.

        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_sample = self.__images[idx]
        if (self.__transform):
            image_sample = self.__transform(image_sample)
        image_sample = image_sample.astype(np.float32)
        return torch.from_numpy(image_sample)

    #  @property
    # def images(self):
    #     """Getter para el atributo 'images'."""
    #     return self.__images

    # @property
    # def patient(self):
    #     """Getter para el atributo 'patient'."""
    #     return self.__patient

    # @property
    # def labels(self):
    #     """Getter para el atributo 'patient'."""
    #     return self.__labels


class PatchClassifierDataset():

    def __init__(
            self,
            path_info_file: str,
            dataset_root_directory: str,
            transform=None,
            pickle_save_file: str = "",
            pickle_load_file: str = "",
    ):
        self.__info = pd.read_excel(path_info_file)
        self.__dataset_root_directory = dataset_root_directory
        self.__transform = transform
        self.__labels = np.array

        if pickle_load_file != "":
            with open(pickle_load_file, "rb") as file:
                data = pickle.load(file)

                self.__images = data["__images"]
                self.__patients = data["__patients"]
                self.__labels = data["__labels"]

        else:
            self.__read_images()
            if pickle_save_file != "":
                with open(pickle_save_file, "wb") as file:
                    data = {
                        "__images": self.__images,
                        "__patients": self.__patients,
                        "__labels": self.__labels,
                    }

                    pickle.dump(data, file)

    def __read_images(self):
        self.__images, self.__patients, self.__labels = load_annotated_patients(
            self.__dataset_root_directory,
            self.__info
        )

    def __len__(self):
        """
        Get amount of images loaded.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return len(self.__images)

    def __getitem__(self, idx):
        """
        Get item method.

        Parameters
        ----------
        idx : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.
        label_sample : TYPE
            DESCRIPTION.

        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_sample = self.__images[idx]
        label_sample = self.__labels[idx]

        if (self.__transform):
            image_sample = self.__transform(image_sample)

        image_sample = image_sample.astype(np.float32)

        return torch.from_numpy(image_sample), label_sample

    #  @property
    # def images(self):
    #     """Getter para el atributo 'images'."""
    #     return self.__images

    # @property
    # def patient(self):
    #     """Getter para el atributo 'patient'."""
    #     return self.__patient

    # @property
    # def labels(self):
    #     """Getter para el atributo 'patient'."""
    #     return self.__labels


def create_dataloaders(class_dataset, batch):
    return DataLoader(class_dataset, batch_size=batch, shuffle=True)
