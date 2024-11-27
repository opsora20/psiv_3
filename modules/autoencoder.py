# -*- coding: utf-8 -*- noqa
"""
Created on Tue Nov 12 12:44:33 2024

@author: joanb
"""

import numpy as np
import torch
from torch import nn as nn
import torch.nn.functional as F
from torch import Tensor

from models_init import init_weights_xavier_normal

# Convolutional


class _CNNLayer(nn.Module):
    def __init__(
        self, num_input_features: int, n_neurons: int, kernel_sze: int = 3,
        stride: int = 1,
        drop_rate: float = 0,
        Relu=True
    ) -> None:
        super().__init__()

        norm1 = nn.BatchNorm2d(n_neurons)
        conv1 = nn.Conv2d(num_input_features, n_neurons, kernel_size=kernel_sze,
                          stride=stride, padding=(int((kernel_sze-1)/2)))

      #  relu1 = nn.ReLU(inplace=True)
        relu1 = nn.LeakyReLU(inplace=True)

        drop = nn.Dropout(drop_rate)
        if Relu:
            self.cnn_layer = nn.Sequential(conv1, norm1, relu1, drop)
        else:
            self.cnn_layer = nn.Sequential(conv1, norm1, drop)
        init_weights_xavier_normal(self)

    def forward(self, x: Tensor):

        return (self.cnn_layer(x))


class _UnCNNLayer(nn.Module):
    def __init__(
        self, num_input_features: int, n_neurons: int, kernel_sze: int = 3,
        stride: int = 2,
        drop_rate: float = 0,
        Relu=True
    ) -> None:
        super().__init__()

        self.stride = stride
        norm1 = nn.BatchNorm2d(n_neurons)
        conv1 = nn.ConvTranspose2d(num_input_features, n_neurons, kernel_size=kernel_sze,
                                   stride=stride, padding=(int((kernel_sze-1)/2)))

     #   relu1 = nn.ReLU(inplace=True)
        relu1 = nn.LeakyReLU(inplace=True)

        drop = nn.Dropout(drop_rate)
        if Relu:
            self.cnn_layer = nn.Sequential(conv1, norm1, relu1, drop)
        else:
            self.cnn_layer = nn.Sequential(conv1, norm1, drop)
        init_weights_xavier_normal(self)

    def forward(self, x: Tensor):

        if self.stride > 1:
            sze_enc = x.shape[-1]
            x = self.cnn_layer[0](x, output_size=(sze_enc*2, sze_enc*2))
            for k in np.arange(1, len(self.cnn_layer)):
                x = self.cnn_layer[k](x)
        else:
            x = self.cnn_layer(x)

        return (x)

    # def forward(self, x1, x2):
    #     x1 = self.cnn_layer(x1)
    #     # input is CHW
    #     diffY = x2.size()[2] - x1.size()[2]
    #     diffX = x2.size()[3] - x1.size()[3]

    #     x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
    #                     diffY // 2, diffY - diffY // 2])
    #     # if you have padding issues, see
    #     # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
    #     # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
    #     x = torch.cat([x2, x1], dim=1)
    #     return self.conv(x)


class _CNNBlock(nn.ModuleDict):
    _version = 2

    def __init__(
        self,
        num_input_channels: int = 1,
        drop_rate=0,
        block_config=(64, 128),
        stride=None,
        decoder=False,
        Relu=True

    ) -> None:
        super().__init__()

        num_layers = len(block_config)
        self.num_input_channels = num_input_channels

        if stride is None:
            stride = np.ones(num_layers)

        for i in range(num_layers):
            if decoder == True:
                layer = _UnCNNLayer(
                    num_input_channels,
                    n_neurons=block_config[i],
                    stride=stride[i],
                    drop_rate=drop_rate

                )
            else:
                layer = _CNNLayer(
                    num_input_channels,
                    n_neurons=block_config[i],
                    stride=stride[i],
                    drop_rate=drop_rate,
                    Relu=Relu

                )
            self.add_module("cnnlayer%d" % (i + 1), layer)
            num_input_channels = block_config[i]

    def forward(self, x: Tensor) -> Tensor:

        for name, layer in self.items():
            x = layer(x)

        return x


#  BACKBONE MODULES
class Encoder(nn.Module):
    r"""Encoder class
    `".
    Input Parameters:
        1. inputmodule_params: dictionary with keys ['num_input_channels']
            inputmodule_params['num_input_channels']=Channels of input images
        2. net_params: dictionary defining architecture: 
            net_params['block_configs']: list of number of neurons for each 
            convolutional block. A block can have more than one layer
            net_params['stride']:list of strides for each block layers
            net_params['drop_rate']: value of the Dropout (equal for all blocks)
        Examples: 
            1. Encoder with 4 blocks with one layer each
            net_params['block_configs']=[[32],[64],[128],[256]]
            net_params['stride']=[[2],[2],[2],[2]]
            2. Encoder with 2 blocks with two layers each
            net_params['block_configs']=[[32,32],[64,64]]
            net_params['stride']=[[1,2],[1,2]]

    """

    def __init__(self, inputmodule_params, net_params):
        super().__init__()

        num_input_channels = inputmodule_params['num_input_channels']

        drop_rate = net_params['drop_rate']
        block_configs = net_params['block_configs'].copy()
        n_blocks = len(block_configs)
        if 'stride' in net_params.keys():
            stride = net_params['stride']
        else:
            stride = []
            for i in np.arange(len(block_configs)):
                stride.append(
                    list(np.ones(len(block_configs[i])-1, dtype=int))+[2])

        # Encoder
        self.encoder = nn.Sequential(
        )
        outchannels_encoder = []
        for i in np.arange(n_blocks):
            block = _CNNBlock(
                num_input_channels=num_input_channels,
                drop_rate=drop_rate,
                block_config=block_configs[i],
                stride=stride[i]

            )
            self.encoder.add_module("cnnblock%d" % (i + 1), block)

            if stride == 1:
                self.encoder.add_module("mxpool%d" % (i + 1),
                                        nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

            num_input_channels = block_configs[i][-1]
           # outchannels_encoder.append(num_input_channels)

    def forward(self, x: Tensor) -> Tensor:

        x = self.encoder(x)

        return x


class Decoder(nn.Module):
    r"""Decoder class
    `".
    Input Parameters:
        1. inputmodule_params: dictionary with keys ['num_input_channels']
            inputmodule_params['num_input_channels']=Channels of input images
        2. net_params: dictionary defining architecture: 
            net_params['block_configs']: list of number of neurons for each conv block
            net_params['stride']:list of strides for each block layers
            net_params['drop_rate']: value of the Dropout (equal for all blocks)
    """

    def __init__(self, inputmodule_params, net_params):
        super().__init__()

        num_input_channels = inputmodule_params['num_input_channels']

        self.upPoolMode = 'bilinear'

        drop_rate = net_params['drop_rate']
        block_configs = net_params['block_configs'].copy()
        self.n_blocks = len(block_configs)

        if 'stride' in net_params.keys():
            stride = net_params['stride']
        else:
            stride = []
            for i in np.arange(len(block_configs)):
                stride.append(
                    list(np.ones(len(block_configs[i])-1, dtype=int))+[2])

        # Decoder
        self.decoder = nn.Sequential(
        )

        for i0 in np.arange(self.n_blocks)[::-1]:
            i = self.n_blocks-(i0+1)
            block = _CNNBlock(
                num_input_channels=num_input_channels,
                drop_rate=drop_rate,
                block_config=block_configs[i],
                stride=stride[i],
                decoder=True
            )

            # if stride==1:
            #     self.decoder.add_module("uppool%d" % (i + 1),
            #                               nn.Upsample(scale_factor=2,
            #                                           mode=self.upPoolMode, align_corners=True))

            self.decoder.add_module("cnnblock%d" % (i0+1), block)

            num_input_channels = block_configs[i][-1]

        self.decoder[-1][list(self.decoder[-1].keys())[-1]
                         ].cnn_layer[2] = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:

        input_sze = x.shape

     #   for i in np.arange(n_blocks)[::-1]:

        x = self.decoder(x)

        return x

# GENERATIVE MODELS


class AutoEncoderCNN(nn.Module):
    r"""AutoEncoderCNN model class
    `".
    Input Parameters:
        1. inputmodule_paramsEnc: dictionary with keys ['num_input_channels']
            inputmodule_paramsEnc['num_input_channels']=Channels of input images
        2. net_paramsEnc: dictionary defining architecture of the Encoder (see Encoder class) 
        3. inputmodule_paramsDec: dictionary with keys ['num_input_channels']
           inputmodule_paramsDec['num_input_channels']=Channels of input images
        4. net_paramsDec: dictionary defining architecture of the Encoder (see Decoder/Encoder classes) 
    """

    def __init__(self, inputmodule_paramsEnc, net_paramsEnc, inputmodule_paramsDec, net_paramsDec):
        super().__init__()

        self.inputmodule_paramsEnc = inputmodule_paramsEnc
        self.inputmodule_paramsDec = inputmodule_paramsDec
        self.net_paramsEnc = net_paramsEnc
        self.net_paramsDec = net_paramsDec
        # Encoder

        self.encoder = Encoder(inputmodule_paramsEnc, net_paramsEnc)

        # Decoder
        self.decoder = Decoder(inputmodule_paramsDec, net_paramsDec)

    def forward(self, x: Tensor) -> Tensor:

        input_sze = x.shape

        x = self.encoder(x)
        x = self.decoder(x)
       # x=F.upsample(x,size=input_sze[2::],mode=self.upPoolMode)

        return x

    def get_embeddings(self, tensor: Tensor, output_size: list) -> Tensor:
        """
        match len(output_size):
            case 1:
                m = nn.AdaptiveAvgPool1d(output_size[0])
            case 2:
                m = nn.AdaptiveAvgPool2d(output_size)
            case 3:
                m = nn.AdaptiveAvgPool3d(output_size)
        """
        if (len(output_size) == 1):
            aux = output_size.copy()
            aux.append(1)
        tensor = self.encoder(tensor)
        aux[0] = aux[0]//(tensor.shape[1])
        m = nn.AdaptiveAvgPool2d(aux)
        pooled = m(tensor)
        pooled = pooled.squeeze(3).view(tensor.size(0), -1)
        return pooled


def AEConfigs(Config):
    inputmodule_paramsEnc = {}
    net_paramsEnc = {}
    inputmodule_paramsDec = {}
    net_paramsDec = {}

    inputmodule_paramsEnc['num_input_channels'] = 3

    net_paramsEnc['drop_rate'] = 2
    net_paramsDec['drop_rate'] = 2

    if Config == '1':
        # CONFIG1
        net_paramsEnc['block_configs'] = [[32, 32], [64, 64]]
        net_paramsEnc['stride'] = [[1, 2], [1, 2]]
        net_paramsDec['block_configs'] = [[64, 32], [
            32, inputmodule_paramsEnc['num_input_channels']]]
        net_paramsDec['stride'] = net_paramsEnc['stride']
        inputmodule_paramsDec['num_input_channels'] = net_paramsEnc['block_configs'][-1][-1]

    elif Config == '2':
        # CONFIG 2
        net_paramsEnc['block_configs'] = [[32], [64], [128], [256]]
        net_paramsEnc['stride'] = [[2], [2], [2], [2]]
        net_paramsDec['block_configs'] = [[128], [64], [32],
                                          [inputmodule_paramsEnc['num_input_channels']]]
        net_paramsDec['stride'] = net_paramsEnc['stride']
        inputmodule_paramsDec['num_input_channels'] = net_paramsEnc['block_configs'][-1][-1]

    elif Config == '3':
        # CONFIG3
        net_paramsEnc['block_configs'] = [[32], [64], [64]]
        net_paramsEnc['stride'] = [[1], [2], [2]]
        net_paramsDec['block_configs'] = [[64], [32], [
            inputmodule_paramsEnc['num_input_channels']]]
        net_paramsDec['stride'] = net_paramsEnc['stride']
        inputmodule_paramsDec['num_input_channels'] = net_paramsEnc['block_configs'][-1][-1]

    return inputmodule_paramsEnc, net_paramsEnc, inputmodule_paramsDec, net_paramsDec
