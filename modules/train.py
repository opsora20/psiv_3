import time
import copy
import torch
import matplotlib.pyplot as plt
import wandb


def train_autoencoder(model, batch_size, loss_func, device, loader, optimizer, num_epochs):
    
    for epoch in range(1, num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        
        t0 = time.time()
        
        train_epoch(model, batch_size, )
        
        
        



def train_epoch(model, batch_size, loss_func, device, loader, optimizer):
    
    model.train()
    
    running_loss = 0.0
    batch_acc = 0.0
    
    for batch_idx, inputs in enumerate(loader):
        inputs = inputs.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss.backward()
        optimizer.step()