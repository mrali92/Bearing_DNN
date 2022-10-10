"""
Created on 15.07.20
@author :ali
"""
import os
from copy import deepcopy
from pathlib import Path
import re

import numpy as np
import torch
from torch import nn, nn as nn

from ml.utils.file_managment import Tee

_layer_types_dict = {"linear": nn.Linear, "bilinear": nn.Bilinear}


class Feed_Forward_Net(nn.Module):
    """Creates a feed-forward network by stacking linearly fully-connected layers."""

    def __init__(self, net_size=[1, 1, 1], functionals=nn.ReLU(), layer_type=torch.nn.Linear, skip_connection=False,
                 batch_norm=False, eps=1e-05, momentum=0.1, dropout=[1, 1]):
        """

       :param net_size:The network dimension. The length of the array refers to number of layers including input and
                        output layer. The element of the array is the number of units per layer
       :type net_size: list
       :param functionals: The activation function applied to all layers. Use a list of activation functions to apply
        them layer wise. (Default value = torch.ReLU())
        :type functionals: Any
       :param layer_type : Type of the transformation applied to all layers to incoming data. Use a list of layer_type
        to apply them layer wise.(Default value = torch.Linear)
       :type layer_type: list or torch.nn.Module
       :param skip_connection : add a skip connection between between two layers after jumping one layer.
       :type skip_connection : bool or list of bool
       :param batch_norm: applies batch normalization over the hidden layer.Use a list of batch_norm to apply them layer
        wise. The length of the batch_norm list should match the number of the hidden layer . (Default value = False)
       :type batch_norm: bool or list of bool
       :param eps: a value added to the denominator for numerical stability. Used when batch normalization is applied.
        (Default value = 1e-5).
       :type eps: float
       :param momentum: the value used for the running_mean and running_var computation.
        Can be set to ``None`` for cumulative moving average (i.e. simple average). Used when batch
       normalization is applied. (Default value = 0.1)
       :type momentum: float
        """

        super(Feed_Forward_Net, self).__init__()

        self.net_size = list(net_size)
        self.eps = eps
        self.momentum = momentum
        self.dropout = dropout

        # input, hidden and output layer
        if type(layer_type) is list:
            self.layer_types = list(layer_type)
        else:
            self.layer_types = []
            for k in range(1, len(net_size)):
                self.layer_types.append(layer_type)

        if type(skip_connection) is list:
            self.skip_connections = [False] + skip_connection
        else:
            self.skip_connections = [False]
            for k in range(1, len(net_size)):
                self.skip_connections.append(skip_connection)

        if type(batch_norm) is list:
            self.batch_norm = [False] + batch_norm + [False]
        else:
            self.batch_norm = [False]  # input layer
            for k in range(1, len(net_size) - 1):
                self.batch_norm.append(batch_norm)
            self.batch_norm.append(False)  # output layer

        bn_layers = nn.ModuleList()
        for k in range(len(net_size) - 1):
            if self.batch_norm[k]:
                bn_layers.append(nn.BatchNorm1d(net_size[k + 1], self.eps, self.momentum))
            else:
                bn_layers.append(nn.Identity(net_size[k + 1]))
        self.bn_layers = bn_layers

        dropout_layers = nn.ModuleList()
        for k in range(len(dropout)):
            dropout_layers.append(nn.Dropout(dropout[k]))
        self.dropout_layers = dropout_layers

        layer = nn.ModuleList()
        for k in range(1, len(net_size)):
            input_size = net_size[k - 1]

            if self.layer_types[k - 1] == torch.nn.Linear:
                layer.append(nn.Linear(input_size, net_size[k]))
            else:
                layer.append(nn.Linear(input_size, net_size[k]))
            if self.layer_types[k - 1] == torch.nn.Bilinear:
                layer.append(nn.Bilinear(input_size, input_size, net_size[k]))
        self.layers = layer

        if not isinstance(functionals, list) and not isinstance(functionals, tuple):
            functionals = (len(net_size) - 1) * [deepcopy(functionals)]
        # TODO: make functionals a module list
        # self.functionals = nn.ModuleList(functionals)
        self.functionals = list(functionals)

    def forward(self, x, a=None):
        """Defines the forward pass performed at each call.

        :param x: The input data
        :type x: Tensor
        :param a: The action layer (Default value = None)
        :type a: Tensor

        """

        layers = self.layers
        bn_layers = self.bn_layers
        functionals = self.functionals
        layer_types = self.layer_types
        skip_cons = self.skip_connections
        dp_layers = self.dropout_layers
        x_prev = []
        for k, (layer, bn_layer, functional, layer_type, skip_con, dp_layer) in enumerate(
                zip(layers, bn_layers, functionals, layer_types, skip_cons, dp_layers)):

            x_prev.append(x)
            if layer_type == nn.Linear:
                x = layer(x)
            elif layer_type == nn.Bilinear:
                x = layer(x, x)
            if skip_con:
                x = x + x_prev[-2]

            # features must be in dim 1 for batch normalization, but layers output is dim=[N,*,features]
            x = bn_layer(x.flatten(0, max(-2, -x.dim()))).reshape(x.shape)

            if functional is not None:
                x = functional(x)
                x = dp_layer(x)
        return x

    def init_weight_and_bias(self, weight_fun=torch.nn.init.normal_,
                             bias_fun=lambda x: torch.nn.init.constant_(x, 0)):
        """Initializes the weights an bias with customized functions

        :param weight_fun: Initialization function for weights(Default value = torch.torch.init.normal_)
        :param bias_fun: Initialization function for bias(Default value = lambda x: torch.torch.init.constant_(x,0)
        :param 0:
        :param 0):

        """

        for layer in self.layers:
            if layer.bias is not None and bias_fun is not None:
                bias_fun(layer.bias)
            if weight_fun is not None:
                weight_fun(layer.weight)

    def load_best_model(self, path):
        run_path = Path(path)
        pattern = "best_model" + "*.pt"
        regex = r"\d+\.\d+"
        best_models_path = []
        for file in run_path.glob(pattern):
            best_models_path.append(file.name)
        best_model_name = [file for file in best_models_path if max([re.findall(regex, i)[0] for i in best_models_path
                                                                     if re.findall(regex, i)[0]]) in file][0]
        best_model_path = os.path.join(run_path, best_model_name)
        self.load_state_dict(torch.load(best_model_path))


def create_ffn_net(net_size, activations, layer_types, skip_connections, weight_init=["normal_", {"std": 0.5}],
                   bias_init="normal_", batch_norm=False,
                   eps=1e-05, momentum=0.1, dropout=None):
    """Creates a Simple Network model.

    :param net_size: The network dimension. The length of the array refers to number of layers including input and
        output layer. The element of the array is the number of units per layer
    :type net_size: list
    :param activations: String or list of strings (or modules respectively) defining the activation functions.
    Supported strings: tanh, relu, none
    :param layer_type: Type of the transformation applied to all layers to incoming data. Use a list of layer_type
        to apply them layer wise.(Default value = torch.Linear)
    :type layer_type: list or torch.nn.Module
    :param skip_connection: add a skip connection between between two layers after jumping one layer.
    :type skip_connection: bool or list of bool
    :param batch_norm: applies batch normalization over the hidden layer.Use a list of batch_norm to apply them layer
        wise. The length of the batch_norm list should match the number of the hidden layer . (Default value = False)
    :type batch_norm: bool or list of bool
    :param eps: a value added to the denominator for numerical stability. Used when batch normalization is applied.
        (Default value = 1e-5).
    :type eps: float
    :param momentum: the value used for the running_mean and running_var computation.
         Can be set to ``None`` for cumulative moving average (i.e. simple average). Used when batch normalization is
         applied. (Default value = 0.1)
    :type momentum: float



    """

    if isinstance(activations, str):
        activations = _get_activation(activations)
    else:
        activations = [_get_activation(activation) if activation else None for activation in activations]

    if isinstance(layer_types, str):
        layer_types_fn = _layer_types_dict[layer_types]
    else:
        layer_types_fn = [_layer_types_dict[layer_type] for layer_type in layer_types]

    if isinstance(dropout, list):
        if len(dropout) >= len(net_size):
            raise RuntimeError("The dropout length mismatch the layer network")
        else:
            dropout = [p for p in dropout if 0 <= p <= 1]
    if isinstance(dropout, float):
        # TODO in elegant manner
        dropout = [float(dropout)] + [0] * len(net_size - 2)

    model = Feed_Forward_Net(net_size, functionals=activations, layer_type=layer_types_fn,
                             skip_connection=skip_connections, batch_norm=batch_norm, eps=eps,
                             momentum=momentum, dropout=dropout)

    weight_init_fn = _get_init_fun(weight_init)
    bias_init_fn = _get_init_fun(bias_init)
    model.init_weight_and_bias(weight_init_fn, bias_init_fn)
    return model


def _get_activation(s):
    if s.lower() == "tanh":
        return torch.nn.Tanh()
    elif s.lower() == "relu":
        return torch.nn.ReLU()
    elif s.lower() == "none":
        return torch.nn.Identity()
    elif s.lower() == "lrelu":
        return torch.nn.LeakyReLU()
    elif s.lower() == "sig":
        return torch.nn.Sigmoid()
    elif s.lower() == "log_softmax":
        return torch.nn.LogSoftmax(dim=-1)
    elif s.lower() == "softmax":
        return torch.nn.Softmax(dim=-1)
    else:
        raise RuntimeError("Unsupported activation, got {:}".format(s))


def _get_init_fun(init):
    """

    :param init:

    """
    if isinstance(init, str):
        init_fn = getattr(torch.nn.init, init)
    elif isinstance(init, list):
        init_fn_noargs = getattr(torch.nn.init, init[0])

        def init_fn(x):
            return init_fn_noargs(x, **init[1])
    else:
        raise TypeError("Expectet input arg to be of type str or list, got {:} instead.".format(type(init)))
    return init_fn


class ConvNet(nn.Module):
    """Convoultion neural network"""

    def __init__(self, convNet_type="Encoder", conv_channels=[1, 1, 1], conv_kernel_size=[3, 2], conv_stride=[1, 1],
                 conv_padding=[0, 0], conv_out_padding=[0, 0], fc_net_size=[5, 5, 1], functionals=nn.ReLU(),
                 batch_norm=False, eps=1e-05,
                 momentum=0.1, dropout=[1, 1], input_dim=10):

        super(ConvNet, self).__init__()
        self.convNet_type = convNet_type
        self.conv_channels = conv_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_stride = conv_stride
        self.conv_padding = conv_padding
        self.conv_out_padding = conv_out_padding
        self.input_dim = input_dim
        self.eps = eps
        self.momentum = momentum
        self.dropout = dropout
        self.fc_net_size = fc_net_size

        self.net_size = len(self.conv_channels) + len(self.fc_net_size)

        if convNet_type == "Encoder":
            # convolution layers
            conv_layers = nn.ModuleList()
            for k in range(len(conv_kernel_size)):
                conv_layers.append(nn.Conv1d(conv_channels[k], conv_channels[k + 1], kernel_size=conv_kernel_size[k],
                                             stride=conv_stride[k], padding=conv_padding[k]))
            self.conv_layers = conv_layers

        elif convNet_type == "Decoder":
            # Deconvolution layers
            conv_layers = nn.ModuleList()
            for k in range(len(conv_kernel_size)):
                conv_layers.append(
                    nn.ConvTranspose1d(conv_channels[k], conv_channels[k + 1], kernel_size=conv_kernel_size[k],
                                       stride=conv_stride[k], padding=conv_padding[k],
                                       output_padding=conv_out_padding[k]))
            self.conv_layers = conv_layers

        conv_net_dims = self.get_output_layers_dims()
        fc_input_dim = conv_net_dims[-1][1] * conv_net_dims[-1][2]

        # fully connected layers
        fc_net_size = [fc_input_dim] + self.fc_net_size
        # input, hidden and output layer
        fc_layers = nn.ModuleList()
        for k in range(1, len(fc_net_size)):
            fc_layers.append(nn.Linear(fc_net_size[k - 1], fc_net_size[k]))
        self.fc_layers = fc_layers

        # activation function
        functionals_list = [[], []]
        for type in range(len(functionals)):
            functionals_list[type] = [_get_activation(activation) if activation else None for activation in
                                      functionals[type]]
        self.functionals = functionals_list

        # self.batch_norm = [[False] + batch_norm[0],  batch_norm[1] + [False]]

        # batch norm layers TODO : fix the redundance
        bn_layers_conv = nn.ModuleList()
        bn_layers_fc = nn.ModuleList()
        net_dims = self.get_net_dims()
        for layer in batch_norm[0]:
            if layer:
                bn_layers_conv.append(nn.BatchNorm1d(1, self.eps, self.momentum))
            else:
                bn_layers_conv.append(nn.Identity(1))
        self.bn_layers_conv = bn_layers_conv

        for k, layer in enumerate(batch_norm[1]):
            if layer:
                bn_layers_fc.append(nn.BatchNorm1d(net_dims[1][k], self.eps, self.momentum))
            else:
                  bn_layers_fc.append(nn.Identity(net_dims[1][k]))
        self.bn_layers_fc = bn_layers_fc

        # dropout layers
        dropout_layers_conv = nn.ModuleList()
        dropout_layers_fc = nn.ModuleList()
        for k in dropout[0]:
            dropout_layers_conv.append(nn.Dropout(k))
        self.dropout_layers_conv = dropout_layers_conv

        for k in dropout[1]:
            dropout_layers_fc.append(nn.Dropout(k))
        self.dropout_layers_fc = dropout_layers_fc

    def forward(self, x):

        conv_layers = self.conv_layers
        bn_layers_conv = self.bn_layers_conv
        bn_layers_fc = self.bn_layers_fc
        functionals = self.functionals
        dropout_layers_conv = self.dropout_layers_conv
        dropout_layers_fc = self.dropout_layers_fc
        fc_layers = self.fc_layers

        for k, (conv_layer, bn_layer, functional, dp_layer) in enumerate(
                zip(conv_layers, bn_layers_conv, functionals[0], dropout_layers_conv)):
            x = conv_layer(x)
            x = bn_layer(x)
            x = functional(x)
            x = dp_layer(x)

        x = torch.flatten(x, 1)

        for k, (fc_layer, bn_layer, functional, dp_layer) in enumerate(
                zip(fc_layers, bn_layers_fc, functionals[1], dropout_layers_fc)):
            x = fc_layer(x)
            x = bn_layer(x)
            # x = bn_layer(x.flatten(0, max(-2, -x.dim()))).reshape(x.shape)
            x = functional(x)
            x = dp_layer(x)
        return x

    def print_dimensions(self):
        net_dims = self.get_output_layers_dims(self.input_dim)
        for k in range(len(net_dims) - 1):
            print("Sequential Layer {:} input  dimension: \tm x {:} x {:}".format(k, net_dims[k][1], net_dims[k][2]))
        print("Net output dimension: \t\t\tm x {:} x {:}".format(net_dims[k + 1][1], net_dims[k + 1][2]))

    def get_net_dims(self):
        conv_dims = np.array(self.get_output_layers_dims())[:, np.newaxis].squeeze()[:, -1]
        fc_dims = np.array(self.fc_net_size)
        return [conv_dims, fc_dims]

    def get_output_layers_dims(self):
        """

        :param input_dim:

        """
        net_dims = []
        conv_layers = self.conv_layers
        # layer_types = self.layer_types
        input_size = self.input_dim
        if self.convNet_type == "Encoder":
            if conv_layers is not None:
                for conv_layer in conv_layers:
                    conv_module = list(conv_layer.modules())[0]

                    output_dim = int(np.floor(
                        ((input_size + 2 * conv_module.padding[0] - conv_module.kernel_size[0]) / conv_module.stride[
                            0]) + 1))
                    net_dims.append([None, conv_module.in_channels, input_size])
                    input_size = output_dim

                net_dims.append([None, conv_module.out_channels, input_size])

        elif self.convNet_type == "Decoder":
            if conv_layers is not None:
                for conv_layer in conv_layers:
                    conv_module = list(conv_layer.modules())[0]

                    output_dim = int(np.floor(
                        (((input_size - 1) * conv_module.stride[0]) - 2 * conv_module.padding[0] +
                         conv_module.kernel_size[0] + conv_module.output_padding[0])))

                    net_dims.append([None, conv_module.in_channels, input_size])
                    input_size = output_dim

                net_dims.append([None, conv_module.out_channels, input_size])

        return net_dims

    def init_weight_and_bias(self, weight_fun, bias_fun):
        """

        :param weight_fun: param bias_fun:
        :param bias_fun:

        """
        if self.conv_layers is not None:
            for conv_layers in self.conv_layers:
                module_list = list(conv_layers.modules())
                conv = module_list[1]
                if conv.bias is not None and bias_fun is not None:
                    bias_fun(conv.bias)
                if weight_fun is not None:
                    weight_fun(conv.weight)
        if self.fc_net_size is not None:
            for fc_layer in self.fc_layers:
                if fc_layer.bias is not None and bias_fun is not None:
                    bias_fun(fc_layer.bias)
                if weight_fun is not None:
                    weight_fun(fc_layer.weight)


class ConvAutoencoder(nn.Module):
    def __init__(self,conv_channels_en,
                conv_kernel_size_en,
                conv_stride_en,
                conv_padding_en,
                conv_out_padding_en,
                conv_channels_de,
                conv_kernel_size_de,
                conv_stride_de,
                conv_padding_de,
                conv_out_padding_de,
                 bottleneck_layer,
                 output_layer,
                 en_functionals,
                 en_batch_norm,
                 en_dropout,
                 de_functionals,
                 de_batch_norm,
                 de_dropout,
                 eps,
                 momentum,
                 input_dim):
        super(ConvAutoencoder, self).__init__()
        self.encoder = ConvNet(convNet_type="Encoder",
                               conv_channels=conv_channels_en,
                               conv_kernel_size=conv_kernel_size_en,
                               conv_stride=conv_stride_en,
                               conv_padding=conv_padding_en,
                               conv_out_padding=conv_out_padding_en,
                               fc_net_size=bottleneck_layer,
                               functionals=en_functionals,
                               batch_norm=en_batch_norm,
                               eps=eps,
                               momentum=momentum,
                               dropout=en_dropout,
                               input_dim=input_dim)

        decoder_input_dim = bottleneck_layer[-1]

        self.decoder = ConvNet(convNet_type="Decoder",
                               conv_channels=conv_channels_de,
                               conv_kernel_size=conv_kernel_size_de,
                               conv_stride=conv_stride_de,
                               conv_padding=conv_padding_de,
                               conv_out_padding=conv_out_padding_de,
                               fc_net_size=output_layer,
                               functionals=de_functionals,
                               batch_norm=de_batch_norm,
                               eps=eps,
                               momentum=momentum,
                               dropout=de_dropout,
                               input_dim=decoder_input_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = x.unsqueeze(1)  # add channel to decoder input
        x = self.decoder(x)
        return x


def weights_init(m, init_type="uniform", gain=1.):
    """initialises weights with selected initialization function and bias with zero.

    Can only be applied to linear, bilinear and conv1d layers. To enable full commandline compatibility and dicts are
    not supported via command line, kwargs must be passed as a list and are redirected as positional arguments.
    Have a look at the torchup documentation in torch.torch.* to get the order of keyword arguments and pass them as a
    list.

    :example:

    :param m: model layer that contains trained parameters (weights, bias).
    :type m: torch.nn
    :param init_type: type of the distribution used in the initialization. Can be "uniform", "normal", "xavier_uniform",
    "xavier_normal", "kaiming_uniform" or "kaiming_normal" corresponding to the init functions from "torch.torch.init.*"
    In addition, init types 'uniform' and 'normal' are scaled with 1/sqrt(fan_in). (Default value = "uniform").
    :type init_type: str
    :param gain: Gain or std used in the initialization function, or (-gain,gain) in 'uniform' function or
    nonlinearity in kaiming function
    :type gain: list: list: list

    >>> tensor = torch.torch.Linear(3,3)
    >>> init_type = "uniform"
    # uniform accepts two keyword arguments (a and b).
    >>> kwargs_list = [-1,1] # a and b as list
    >>> weights_init(tensor,init_type, kwargs_list)
    """

    if init_type == "xavier_uniform":
        def init_fun(tensor):
            nn.init.xavier_uniform_(tensor, gain=gain)
    elif init_type == "xavier_normal":
        def init_fun(tensor):
            nn.init.xavier_normal_(tensor, gain=gain)
    elif init_type == "uniform":
        def init_fun(tensor):
            """

            :param tensor: param *gain:
            :param *gain:

            """
            with torch.no_grad():
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(tensor)
                torch.nn.init.uniform_(tensor, a=-gain, b=gain)
                tensor.data = tensor.data / np.sqrt(fan_in)
    elif init_type == "normal":
        def init_fun(tensor):
            """

            :param tensor: param *gain:
            :param *gain:

            """
            with torch.no_grad():
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(tensor)
                torch.nn.init.normal_(tensor, std=gain)
                tensor.data = tensor.data / np.sqrt(fan_in)
    elif init_type == "kaiming_normal":
        def init_fun(tensor):
            torch.nn.init.kaiming_normal_(tensor, nonlinearity=gain)
    elif init_type == "kaiming_uniform":
        def init_fun(tensor):
            torch.nn.init.kaiming_uniform_(tensor, nonlinearity=gain)

    else:
        raise RuntimeError("init_type does not match any valid init type, got {:}. See help for more "
                           "information.".format(init_type))

    if isinstance(m, nn.Linear) or isinstance(m, nn.Bilinear) or isinstance(m, nn.Conv1d):
        init_fun(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)