# -*- coding: utf-8 -*- noqa
"""
Created on Tue Nov 12 12:44:33 2024

@author: joanb
"""

import time
import sys

from copy import deepcopy
from utils import echo


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

        __train_epoch(
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
        model,
        loss_func,
        device,
        loader,
        optimizer,
        num_epochs,
        precission: float = 0.001
):
    
    pass