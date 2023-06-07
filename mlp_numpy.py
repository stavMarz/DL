################################################################################
#
# BGU IEM Introduction to Deep Learning Course | 2023b
#
################################################################################

"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import *
import numpy as np


class MLP(object):
    """
    This class implements a Multi-layer Perceptron in NumPy.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward and backward.
    """

    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes MLP object.

        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model 
                    will simply perform a multinomial logsistic regression.
          n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP

        TODO:
        Implement initialization of the network.
        """
        self.layers = []
        #print("n_hidden",len(n_hidden))
        # Add linear layers
        if len(n_hidden) > 0:

            self.layers.append(LinearModule(n_inputs, n_hidden[0], input_layer=True))
            for i in range(len(n_hidden) - 1):
                #print(i+1)
                self.layers.append(ReluModule())
                self.layers.append(LinearModule(n_hidden[i], n_hidden[i + 1]))
            self.layers.append(ReluModule())
            #print("now",n_hidden[-1])
            self.layers.append(LinearModule(n_hidden[-1], n_classes))

            #print(self.layers)
            #print("len_layer",len(self.layers))

        else:
            self.layers.append(LinearModule(n_inputs, n_classes, input_layer=True))
        self.layers.append(SoftMaxModule())

        
    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network
        Returns:s
          out: outputs of the network

        TODO:
        Implement forward pass of the network.
        """
        out = x  # Initialize the output as the input

        for layer in self.layers:
          out = layer.forward(out)  # Pass the output of the previous layer as input to the current layer

        return out

    def backward(self, dout):
        """
        Performs backward pass given the gradients of the loss.

        Args:
          dout: gradients of the loss with respec to the network output

        TODO:
        Implement backward pass of the network.
        """
        out = dout  # Initialize the output as the input

        for layer in reversed(self.layers):
            out = layer.backward(out)
        
        return out

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass from any module.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Iterate over modules and call the 'clear_cache' function.
        """
        
        for layer in self.layers:
            layer.clear_cache()
