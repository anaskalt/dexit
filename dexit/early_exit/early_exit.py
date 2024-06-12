"""
Early Exit Network Implementation

This module implements an early exit neural network for inference. The model is capable of exiting 
early based on defined thresholds, allowing for faster inference in cases where early predictions 
are confident.

Original Developer: Ilias Paralikas [https://github.com/Ilias-Paralikas]
Contributors: Anastasios Kaltakis

Description:
- `EarlyExitNet`: Main class implementing the early exit logic.
- `EarlyExitNetworkSegmentor`: Class for segmenting the network for early exit.
- Helper functions for temperature scaling and forward pass.

"""

import torch
import torch.nn as nn
import copy

def softmax_temperature(logits, temperature=5):
    """
    Apply temperature scaling to logits.

    Args:
        logits (torch.Tensor): Logits from the model.
        temperature (float): Temperature value for scaling.

    Returns:
        torch.Tensor: Scaled logits.
    """
    return torch.exp(logits / temperature) / torch.sum(torch.exp(logits / temperature))

class EarlyExitNet(nn.Module):
    """
    Early Exit Network with multiple exit points.

    Args:
        network (nn.Module): The base neural network.
        input_shape (tuple): The shape of the input data.
        device (str): The device to run the model on ('cpu' or 'cuda').
        thresholds (list): Thresholds for each exit layer.
        exit_layers (int or list): The number of exit layers or a list specifying exit layers.
        scaler (callable, optional): Scaling function for logits.

    Attributes:
        device (str): The device to run the model on.
        thresholds (list): Thresholds for each exit layer.
        network (nn.Module): The base neural network.
        output_shape (int): The shape of the network output.
        len (int): The length of the network.
        exits (nn.ModuleList): List of exit layers.
        scaler (callable): Scaling function for logits.
    """
    def __init__(self, network, input_shape, device, thresholds, exit_layers=16, scaler=None):
        super(EarlyExitNet, self).__init__()
        self.device = device
        self.thresholds = thresholds
        self.network = network.to(self.device)
        x = torch.rand(input_shape).to(self.device)
        
        if isinstance(exit_layers, int):
            layers = exit_layers
            exit_layers = [[layers] for _ in range(len(network) - 1)]
        
        self.output_shape = network(x).size(1)
        self.len = len(network)
        assert len(exit_layers) == len(network) - 1

        self.exits = nn.ModuleList([])
        for i, net in enumerate(network[:-1]):
            layers = [nn.Flatten()]
            last_layer_size = self._get_output_flattened(net, input_shape)
            for next_layer_size in exit_layers[i]:
                layers.append(nn.Linear(last_layer_size, next_layer_size))
                layers.append(nn.ReLU())
                last_layer_size = next_layer_size
            layers.append(nn.Linear(last_layer_size, self.output_shape))
            input_shape = net(torch.rand(input_shape).to(self.device)).size()
            self.exits.append(nn.Sequential(*layers).to(self.device))
        
        self.scaler = scaler if scaler is not None else softmax_temperature

    def _get_output_flattened(self, network, input_shape):
        """
        Get the flattened output size of a network.

        Args:
            network (nn.Module): The base neural network.
            input_shape (tuple): The shape of the input data.

        Returns:
            int: Flattened output size.
        """
        x = torch.rand(input_shape).to(self.device)
        return network(x).view(x.size(0), -1).size(1)

    def forward(self, x):
        """
        Forward pass through the network with early exits.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            list: List of outputs from each exit layer.
        """
        outputs = []
        for i in range(self.len - 1):
            x = self.network[i](x)
            early_exit = self.exits[i](x)
            early_exit = self.scaler(early_exit)
            outputs.append(early_exit)
        x = self.network[-1](x)
        x = self.scaler(x)
        outputs.append(x)
        return outputs

    def segmented_forward(self, x):
        """
        Forward pass through the network with early exits based on thresholds.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: Output tensor and exit layer index.
        """
        x = x.to(self.device)
        x = x.unsqueeze(0)
        for i in range(self.len - 1):
            x = self.network[i](x)
            early_exit = self.exits[i](x)
            early_exit = early_exit.squeeze(0)
            early_exit = self.scaler(early_exit)
            if early_exit.max() > self.thresholds[i]:
                return early_exit, i
        x = self.network[-1](x)
        x = self.scaler(x)
        return x, i + 1

class EarlyExitNetworkSegmentor(nn.Module):
    """
    Segmentor for the Early Exit Network.

    Args:
        network (nn.Module): The base neural network.
        exit (nn.Module, optional): The exit layer.
        threshold (float, optional): The threshold for early exit.
        device (str): The device to run the model on ('cpu' or 'cuda').
        scaler (callable, optional): Scaling function for logits.

    Attributes:
        network (nn.Module): The base neural network.
        exit (nn.Module): The exit layer.
        threshold (float): The threshold for early exit.
        device (str): The device to run the model on.
        scaler (callable): Scaling function for logits.
    """
    def __init__(self, network, exit=None, threshold=None, device='cpu', scaler=None):
        super(EarlyExitNetworkSegmentor, self).__init__()
        self.network = copy.deepcopy(network)
        self.exit = copy.deepcopy(exit) if exit is not None else None
        self.threshold = threshold
        self.device = device
        self.scaler = scaler

    def forward(self, x):
        """
        Forward pass through the segmentor with early exit.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: Output tensor and boolean indicating if early exit was taken.
        """
        x = x.to(self.device)
        x = x.unsqueeze(0)
        x = self.network(x)
        if self.exit is not None:
            early_exit = self.exit(x)
            early_exit = early_exit.squeeze(0)
            early_exit = self.scaler(early_exit)
            if early_exit.max() > self.threshold:
                return early_exit, True
            return x.squeeze(0), False
        x = x.squeeze(0)
        return x, True
