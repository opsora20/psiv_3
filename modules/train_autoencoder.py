# -*- coding: utf-8 -*- noqa
"""
Created on Tue Nov 12 12:44:33 2024

@author: joanb
"""

import time
import sys

from copy import deepcopy
from utils import echo
import torch
import torch.nn.functional as F
from datasets import create_dataloaders




def train_autoencoder(
        model,
        loss_func,
        device,
        loader,
        optimizer,
        num_epochs,
        precission: float = 0.001
):
    """
    Train autoencoder.

    Parameters
    ----------
    model : TYPE
        DESCRIPTION.
    batch_size : TYPE
        DESCRIPTION.
    loss_func : TYPE
        DESCRIPTION.
    device : TYPE
        DESCRIPTION.
    loader : TYPE
        DESCRIPTION.
    optimizer : TYPE
        DESCRIPTION.
    num_epochs : TYPE
        DESCRIPTION.
    precission : float, optional
        DESCRIPTION. The default is 0.001.

    Returns
    -------
    model : TYPE
        DESCRIPTION.

    """
    loss_log = {x: [] for x in list(loader.keys())}

    best_model_wts = deepcopy(model.state_dict())
    best_loss = sys.maxsize

    for epoch in range(1, num_epochs+1):
        echo(f'Epoch {epoch}/{num_epochs}')
        echo('-' * 10)

        t0 = time.time()

        model, loss_log = __train_epoch(
            model,
            loss_func,
            device,
            loader,
            optimizer,
            loss_log
        )

        epoch_time = time.time() - t0

        if loss_log['train'][-1] < best_loss:
            best_loss = loss_log['train'][-1]
            best_model_wts = deepcopy(model.state_dict())
            best_epoch = epoch

        if (epoch > 2):
            if abs(loss_log['train'][-2] - loss_log['train'][-1]) < precission:
                break

        echo("Epoch elapsed time: {:.4f}s \n".format(epoch_time))

    echo('Best val Loss: {:4f} at epoch {}'.format(best_loss, best_epoch))
    model.load_state_dict(best_model_wts)
    return model


def __train_epoch(
        model,
        loss_func,
        device,
        loader,
        optimizer,
        loss_log
):
    for phase in list(loader.keys()):
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0

        for batch_idx, inputs in enumerate(loader[phase]):

            inputs = inputs.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = loss_func(inputs, outputs)
            if phase == 'train':
                loss.backward()
                optimizer.step()

            running_loss += loss

        epoch_loss = running_loss/len(loader[phase])

        echo(f'{phase} Loss:{epoch_loss:.4f}')
        loss_log[phase].append(epoch_loss)

        return model, loss_log

def train_attention(
        encoder,
        model_att,
        model,
        loss_func,
        device,
        dataset,
        loader,
        patient_dict,
        output_size,
        optimizers,
        num_epochs,
        patient_batch = 1024,
        precission: float = 0.001
):
    
    loss_log = {x: [] for x in list(loader.keys())}

    best_model_wts = deepcopy(model.state_dict())
    best_loss = sys.maxsize
    model.train()
    encoder.eval()
    for epoch in range(1, num_epochs+1):
        echo(f'Epoch {epoch}/{num_epochs}')
        echo('-' * 10)

        t0 = time.time()

        model_att, model, loss_log = __train_epoch_attention(
            encoder,
            model_att, 
            model,
            loss_func,
            device,
            dataset,
            loader,
            patient_dict,
            output_size,
            optimizers,
            loss_log,
            patient_batch
        )

        epoch_time = time.time() - t0

        if loss_log['train'][-1] < best_loss:
            best_loss = loss_log['train'][-1]
            best_model_wts = deepcopy(model.state_dict())
            best_epoch = epoch

        if (epoch > 2):
            if abs(loss_log['train'][-2] - loss_log['train'][-1]) < precission:
                break

        echo("Epoch elapsed time: {:.4f}s \n".format(epoch_time))

    echo('Best val Loss: {:4f} at epoch {}'.format(best_loss, best_epoch))
    model.load_state_dict(best_model_wts)
    return model_att, model


def __train_epoch_attention(
            encoder, 
            model_att,
            model,
            loss_func,
            device,
            dataset,
            loader,
            patient_dict,
            output_size,
            optimizers,
            loss_log, 
            patient_batch = 1024
):
    encoder.eval()
    for phase in list(loader.keys()):
        if phase == 'train':
            model.train()
            model_att.train()
        else:
            model.eval()

        count = 1 
        running_loss = 0.0
        for patient, label in patient_dict.items():
            if (label == 1):
                target = torch.tensor([0.0, 1.0], dtype=torch.float32)
            else:
                target = torch.tensor([1.0, 0.0], dtype=torch.float32)
            process = dataset.load_patient(patient, patient_batch)
            #loader[phase] = create_dataloaders(dataset, batch = 16) 
            if(process):
                for idx, patches in enumerate(loader[phase]):
                    if(patches.shape[0]>1):
                        patches = patches.to(device)
                        patches = encoder.get_embeddings(patches, output_size)
                        Z, A = model_att(patches)
                        preds = model(Z)
                        patient_pred = preds.mean(dim=0)
                        loss = loss_func(patient_pred.cpu(), target)
                        loss.backward()
                        running_loss += loss
                if(count % 16 == 0):
                    optimizers["NN"].step()
                    optimizers["NN"].zero_grad()
                    optimizers["attention"].step()
                    optimizers["attention"].zero_grad()
                count+=1
        if (count % 16) != 0:
            optimizers["NN"].step()
            optimizers["NN"].zero_grad()
            optimizers["attention"].step()
            optimizers["attention"].zero_grad()

        epoch_loss = running_loss/len(patient_dict)
        echo(f'{phase} Loss:{epoch_loss:.4f}')
        loss_log[phase].append(epoch_loss)

        return model_att, model, loss_log