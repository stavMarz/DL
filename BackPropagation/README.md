# BackPropagation

Welcome to the BackPropagation folder! This folder contains a collection of files related to implementing and training a Multi-Layer Perceptron (MLP) using backpropagation algorithm for deep learning tasks. In this folder, you will find the following files:

## Files

### cifar10_utils.py

The `cifar10_utils.py` file contains utility functions for loading and preprocessing the CIFAR-10 dataset. CIFAR-10 is a popular image classification dataset with 60,000 32x32 color images across 10 classes. This file provides functions to download, load, normalize, and split the dataset into training and test sets.

### mlp_numpy.py

The `mlp_numpy.py` file implements an MLP using the NumPy library. This implementation allows you to create a customizable MLP architecture with configurable layers, activation functions, and loss functions. The file provides classes and functions for constructing, training, and evaluating the MLP model using backpropagation.

### train_mlp_numpy.py

The `train_mlp_numpy.py` file demonstrates the usage of the `mlp_numpy.py` module for training an MLP model on the CIFAR-10 dataset. It contains code that sets up the MLP architecture, defines the training loop, and performs the training process. This file showcases how to use the backpropagation algorithm to update the model's weights and biases based on the computed gradients.

### models.py

The `models.py` file contains additional classes that are essential for constructing an MLP model. These classes include the `CrossEntropyLoss` for computing the cross-entropy loss, the `Linear` module for implementing the linear layer of the MLP, the `ReLU` activation function, and the `SoftMaxModule` for obtaining the probability distribution over the classes.

### train_mlp_pytorch.py

The `train_mlp_pytorch.py` file demonstrates an alternative approach to training an MLP model using the PyTorch library. It showcases how to leverage existing PyTorch modules and functions to construct, train, and evaluate an MLP model with backpropagation. This implementation also includes an example of utilizing CUDA for GPU acceleration, if available.

### mlp_torch.py

The `mlp_torch.py` file provides an implementation of the MLP model using PyTorch modules. It defines the architecture of the MLP by stacking linear layers with activation functions, allowing you to create a flexible and customizable MLP model in PyTorch.

## Usage

To use these files, make sure you have the required dependencies installed, including NumPy and PyTorch if you intend to run the PyTorch-based files. The files are well-commented, providing detailed explanations of the code and how to use each component. Feel free to explore and modify the files to suit your specific needs.

## Contributing

This repository is open to contributions. If you have any improvements, bug fixes, or additional functionality that you would like to contribute, please submit a pull request. Your contributions are highly appreciated!
