import torch
from statistics import mean

def MSE_loss(inputs, outputs):
    batch_losses = []
    for input, output in zip(inputs, outputs):
        diff = (input - output)**2
        summed_pixel = diff.sum(dim=2, keepdim=True)
        eucl_dist = torch.sqrt(summed_pixel)
        img_loss =  eucl_dist.mean()
        value = img_loss.item()
        batch_losses.append(value)
    loss = mean(batch_losses)
    return loss

input = torch.randn(5, 5, 3)
output = torch.randn(5, 5, 3)
loss = MSE_loss([input], [output])
print(loss)