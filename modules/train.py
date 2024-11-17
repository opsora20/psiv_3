import time
import copy
import torch
import matplotlib.pyplot as plt
import sys
from utils import echo


def train_autoencoder(model, batch_size, loss_func, device, loader, optimizer, num_epochs):
    
    loss_log = {x:[] for x in list(loader.keys())}
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = sys.maxsize
    
    for epoch in range(1, num_epochs+1):
        echo('Epoch {}/{}'.format(epoch, num_epochs))
        echo('-' * 10)
        
        t0 = time.time()
        
        train_epoch(model, batch_size, loss_func, device, loader, optimizer, loss_log)
        
        epoch_time = time.time() - t0
        
        if loss_log['train'][-1] < best_loss:
            best_loss = loss_log['train'][-1]
            best_epoch = epoch
        
        echo("Epoch elapsed time: {:.4f}s \n".format(epoch_time))
        
        
    echo('Best val Loss: {:4f} at epoch {}'.format(best_loss, best_epoch))
    model.load_state_dict(best_model_wts)
    return model


def train_epoch(model, batch_size, loss_func, device, loader, optimizer, loss_log):
    for phase in list(loader.keys()):
        if phase == 'train':
            model.train()  
        else:
            model.eval()  

        running_loss = 0.0
        batch_acc = 0.0
        
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
        
        echo('{} Loss:{:.4f}'.format(phase, epoch_loss))
        loss_log[phase].append(epoch_loss)
        
        return model, loss_log
    
    