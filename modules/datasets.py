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

from load_datasets import load_cropped_patients, load_annotated_patients, load_patient_images

import os
from skimage import io, color

target_height, target_width = 256, 256


class AutoEncoderDataset(Dataset):

    def __init__(
            self,
            path_info_file: str,
            dataset_root_directory: str,
            transform=None,
            read=True,
            pickle_save_file: str = "",
            pickle_load_file: str = "",
    ):
        self.__info = pd.read_csv(path_info_file)
        self.__dataset_root_directory = dataset_root_directory
        self.__transform = transform
        self.__read = read

        if pickle_load_file != "":
            with open(pickle_load_file, "rb") as file:
                data = pickle.load(file)

                self.__images = data["__images"]

        elif (self.__read):
            self.__read_images()
            if pickle_save_file != "":
                with open(pickle_save_file, "wb") as file:
                    data = {"__images": self.__images}

                    pickle.dump(data, file)
        else:
            self.__read_names()

    def __read_images(self):
        self.__images = load_cropped_patients(
            self.__dataset_root_directory, self.__info)

    def __read_names(self):
        self.__images = []
        for patient_directory in os.listdir(self.__dataset_root_directory):
            aux = patient_directory[:-2]
            dens = self.__info[self.__info["CODI"] == aux]["DENSITAT"].iloc[0]
            if (dens == "NEGATIVA"):
                for image_file in os.listdir(os.path.join(
                        self.__dataset_root_directory,
                        patient_directory
                )):
                    if (image_file.endswith(".png")):
                        self.__images.append(os.path.join(
                            self.__dataset_root_directory,
                            patient_directory,
                            image_file
                        ))
        self.__images = np.array(self.__images)

    def __read_image_sample(self, image_path):
        image = io.imread(image_path)
        image = color.rgba2rgb(image)
        if (image.shape[0] != 256 or image.shape[1] != 256):
            mean_pixel_value = image.mean(axis=(0, 1))
            original_height, original_width, channels = image.shape
            canvas = np.ones(
                (target_height, target_width,
                 channels),
                dtype=image.dtype) * mean_pixel_value

            y_offset = (target_height - original_height) // 2
            x_offset = (target_width - original_width) // 2

            canvas[y_offset:y_offset+original_height,
                   x_offset:x_offset+original_width] = image
            image = canvas

        image = image.transpose(2, 0, 1)
        return image

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
        if (self.__read):

            if (self.__transform):
                image_sample = self.__transform(image_sample)
            image_sample = image_sample.astype(np.float32)
            return torch.from_numpy(image_sample)
        else:
            image_sample = self.__read_image_sample(image_sample)
            if (self.__transform):
                image_sample = self.__transform(image_sample)
            image_sample = image_sample.astype(np.float32)
            return torch.from_numpy(image_sample)

    @property
    def images(self):
        """Getter para el atributo 'images'."""
        return self.__images

    @property
    def patient(self):
        """Getter para el atributo 'patient'."""
        return self.__patient

    @property
    def labels(self):
        """Getter para el atributo 'patient'."""
        return self.__labels


class PatchClassifierDataset(Dataset):

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

    @property
    def images(self):
        """Getter para el atributo 'images'."""
        return self.__images

    @property
    def patients(self):
        """Getter para el atributo 'patient'."""
        return self.__patients

    @property
    def labels(self):
        """Getter para el atributo 'patient'."""
        return self.__labels


class PatientDataset(Dataset):

    def __init__(
            self,
            path_info_file: str,
            dataset_root_directory: str,
            transform=None
    ):
        self.__info = pd.read_csv(path_info_file)
        self.__dataset_root_directory = dataset_root_directory
        self.__transform = transform

    def load_patient(self, patient, max_images):
        self._patient = patient
        self.__images = load_patient_images(patient, self.__dataset_root_directory, max_images)

    def __len__(self):
        return len(self.__info['CODI'])
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_sample = self.__images[idx]

        if (self.__transform):
            image_sample = self.__transform(image_sample)

        image_sample = image_sample.astype(np.float32)

        return torch.from_numpy(image_sample)
    
    @property
    def images(self):
        """Getter para el atributo 'images'."""
        return self.__images


def create_dataloaders(class_dataset, batch):
    return DataLoader(class_dataset, batch_size=batch, shuffle=True)
