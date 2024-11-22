# -*- coding: utf-8 -*- noqa
"""
Created on Tue Nov 12 12:44:33 2024

@author: joanb
"""

import numpy as np

from torch import nn as nn
import torch.nn.functional as F
from torch import Tensor

from models_init import init_weights_xavier_normal


# Convolutional
class _CNNLayer(nn.Module):
    def __init__(
        self,
        num_input_features: int,
        n_neurons: int,
        kernel_sze: int = 3,
        drop_rate: float = 0,
        dim=2,
        Relu=True,
    ) -> None:
        super().__init__()

        if dim == 2:
            norm1 = nn.BatchNorm2d(n_neurons)
            conv1 = nn.Conv2d(
                num_input_features,
                n_neurons,
                kernel_size=kernel_sze,
                stride=(int((kernel_sze-1)/2)),
                padding=(int((kernel_sze-1)/2)),
            )
        elif dim == 3:
            norm1 = nn.BatchNorm3d(n_neurons)
            conv1 = nn.Conv3d(
                num_input_features,
                n_neurons,
                kernel_size=kernel_sze,
                stride=(int((kernel_sze-1)/2)),
                padding=(int((kernel_sze-1)/2)),
            )

        relu1 = nn.ReLU(inplace=True)

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
        self,
        num_input_features: int,
        n_neurons: int,
        kernel_sze: int = 3,
        drop_rate: float = 0,
        dim=2,
        Relu=False,
    ) -> None:
        super().__init__()

        if dim == 2:
            norm1 = nn.BatchNorm2d(n_neurons)
            conv1 = nn.ConvTranspose2d(
                num_input_features,
                n_neurons,
                kernel_size=kernel_sze,
                stride=(int((kernel_sze-1)/2)),
                padding=(int((kernel_sze-1)/2)),
            )
        elif dim == 3:
            norm1 = nn.BatchNorm3d(n_neurons)
            conv1 = nn.ConvTranspose3d(
                num_input_features,
                n_neurons,
                kernel_size=kernel_sze,
                stride=(int((kernel_sze-1)/2)),
                padding=(int((kernel_sze-1)/2)),
            )

        relu1 = nn.ReLU(inplace=True)

        drop = nn.Dropout(drop_rate)

        if Relu:
            self.cnn_layer = nn.Sequential(conv1, norm1, relu1, drop)
        else:
            self.cnn_layer = nn.Sequential(conv1, norm1, drop)

        init_weights_xavier_normal(self)

    def forward(self, x: Tensor):
        return (self.cnn_layer(x))


class _CNNBlock(nn.ModuleDict):
    _version = 2

    def __init__(
        self,
        num_input_channels: int = 1,
        drop_rate=0,
        block_config=(64, 128),
        dim=2,
        decoder=False,
        Relu=True

    ) -> None:
        super().__init__()

        num_layers = len(block_config)
        self.num_input_channels = num_input_channels
        for i in range(num_layers):
            if decoder:
                layer = _UnCNNLayer(
                    num_input_channels,
                    n_neurons=block_config[i],
                    drop_rate=drop_rate, dim=dim

                )
            else:
                layer = _CNNLayer(
                    num_input_channels,
                    n_neurons=block_config[i],
                    drop_rate=drop_rate, dim=dim,
                    Relu=Relu

                )
            self.add_module("cnnlayer%d" % (i + 1), layer)
            num_input_channels = block_config[i]

    def forward(self, x: Tensor) -> Tensor:

        for name, layer in self.items():
            x = layer(x)

        return x


def AEConfigs(Config):
    """
    Get configs for AutoEncoders.

    Parameters
    ----------
    Config : TYPE
        DESCRIPTION.

    Returns
    -------
    inputmodule_paramsEnc : TYPE
        DESCRIPTION.
    net_paramsEnc : TYPE
        DESCRIPTION.
    inputmodule_paramsDec : TYPE
        DESCRIPTION.
    net_paramsDec : TYPE
        DESCRIPTION.

    """
    net_paramsEnc = {}
    net_paramsDec = {}
    inputmodule_paramsEnc = {}
    inputmodule_paramsDec = {}

    inputmodule_paramsEnc['num_input_channels'] = 3

    if Config == '1':
        # CONFIG1
        net_paramsEnc['block_configs'] = [
            [32, 32],
            [64, 64],
        ]

        net_paramsEnc['stride'] = [
            [1, 2],
            [1, 2],
        ]

        net_paramsDec['block_configs'] = [
            [
                inputmodule_paramsEnc['num_input_channels'],
                inputmodule_paramsEnc['num_input_channels'],
            ],
            [32, 32],
        ]

        net_paramsDec['stride'] = net_paramsEnc['stride']

        inputmodule_paramsDec['num_input_channels'] = net_paramsEnc['block_configs'][-1][-1]

        net_paramsEnc['dim'] = 2
        net_paramsDec['dim'] = 2

        net_paramsEnc['drop_rate'] = 0
        net_paramsDec['drop_rate'] = 0

    elif Config == '2':
        # CONFIG 2
        net_paramsEnc['block_configs'] = [
            [32],
            [64],
            [128],
            [256],
        ]

        net_paramsEnc['stride'] = [
            [2],
            [2],
            [2],
            [2],
        ]

        net_paramsDec['block_configs'] = [
            [inputmodule_paramsEnc['num_input_channels']],
            [32],
            [64],
            [128],
        ]

        net_paramsDec['stride'] = net_paramsEnc['stride']

        inputmodule_paramsDec['num_input_channels'] = net_paramsEnc['block_configs'][-1][-1]

        net_paramsEnc['dim'] = 2
        net_paramsDec['dim'] = 2

        net_paramsEnc['drop_rate'] = 0
        net_paramsDec['drop_rate'] = 0

    elif Config == '3':
        # CONFIG3
        net_paramsEnc['block_configs'] = [
            [32],
            [64],
            [64],
        ]

        net_paramsEnc['stride'] = [
            [1],
            [2],
            [2],
        ]
        net_paramsDec['block_configs'] = [
            [inputmodule_paramsEnc['num_input_channels']],
            [32],
            [64],
        ]

        net_paramsDec['stride'] = net_paramsEnc['stride']

        inputmodule_paramsDec['num_input_channels'] = net_paramsEnc['block_configs'][-1][-1]

        net_paramsEnc['dim'] = 2
        net_paramsDec['dim'] = 2

        net_paramsEnc['drop_rate'] = 0
        net_paramsDec['drop_rate'] = 0

    elif Config == '4':
        # CONFIG1
        net_paramsEnc['block_configs'] = [
            [32, 32],
            [64, 64],
            [128, 128],
        ]

        net_paramsEnc['stride'] = [
            [1, 2],
            [1, 2],
            [1, 2],
        ]

        net_paramsDec['block_configs'] = [
            [
                inputmodule_paramsEnc['num_input_channels'],
                inputmodule_paramsEnc['num_input_channels'],
            ],
            [32, 32],
            [64, 64]
        ]
        net_paramsDec['stride'] = net_paramsEnc['stride']

        inputmodule_paramsDec['num_input_channels'] = net_paramsEnc['block_configs'][-1][-1]

        net_paramsEnc['dim'] = 2
        net_paramsDec['dim'] = 2

        net_paramsEnc['drop_rate'] = 0
        net_paramsDec['drop_rate'] = 0

    return inputmodule_paramsEnc, net_paramsEnc, inputmodule_paramsDec, net_paramsDec,


class AutoEncoderCNN(nn.Module):
    def __init__(
            self,
            inputmodule_paramsEnc,
            net_paramsEnc,
            inputmodule_paramsDec,
            net_paramsDec
    ):
        super().__init__()
        self.out_embeddings = inputmodule_paramsDec["num_input_channels"]
        num_input_channels = inputmodule_paramsEnc['num_input_channels']
        self.dim = net_paramsEnc['dim']
        self.upPoolMode = 'bilinear'
        if self.dim == 3:
            self.upPoolMode = 'trilinear'

        drop_rate = net_paramsEnc['drop_rate']
        block_configs = net_paramsEnc['block_configs']
        n_blocks = len(block_configs)

        self.prct = None

        # Encoder
        self.encoder = nn.Sequential(
        )
        for i in np.arange(n_blocks):
            block = _CNNBlock(
                num_input_channels=num_input_channels,
                drop_rate=drop_rate,
                block_config=block_configs[i],
                dim=self.dim

            )
            self.encoder.add_module("cnnblock%d" % (i + 1), block)
            if self.dim == 2:
                self.encoder.add_module(
                    "mxpool%d" % (i + 1),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
                )
            elif self.dim == 3:
                self.encoder.add_module(
                    "mxpool%d" % (i + 1),
                    nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
                )
            num_input_channels = block_configs[i][-1]

        num_input_channels = inputmodule_paramsDec['num_input_channels']
        self.dim = net_paramsDec['dim']
        self.upPoolMode = 'bilinear'
        if self.dim == 3:
            self.upPoolMode = 'trilinear'

        drop_rate = net_paramsDec['drop_rate']
        block_configs = net_paramsDec['block_configs']
        n_blocks = len(block_configs)

        # Decoder
        self.decoder = nn.Sequential(
        )

        for i in np.arange(n_blocks)[::-1]:
            block = _CNNBlock(
                num_input_channels=num_input_channels,
                drop_rate=drop_rate,
                block_config=block_configs[i][::-1],
                dim=self.dim,
                decoder=True
            )
            self.decoder.add_module(
                "uppool%d" % (i + 1),
                nn.Upsample(
                    scale_factor=2,
                    mode=self.upPoolMode,
                    align_corners=True
                )
            )

            self.decoder.add_module("cnnblock%d" % (i + 1), block)

            num_input_channels = block_configs[i][0]

    def forward(self, x: Tensor) -> Tensor:
        """
        Launch forward.

        Parameters
        ----------
        x : Tensor
            DESCRIPTION.

        Returns
        -------
        Tensor
            DESCRIPTION.

        """
        input_sze = x.shape

        x = self.encoder(x)
        x = self.decoder(x)
        x = F.upsample(x, size=input_sze[2::], mode=self.upPoolMode)
        return x

    def get_embeddings(self, x: Tensor) -> Tensor:
        
        x = self.encoder(x)
        print(x.shape)
        x = x.amax(axis=(2, 3))
        print(x.shape)
        return x
