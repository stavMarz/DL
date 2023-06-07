################################################################################
#
# BGU IEM Introduction to Deep Learning Course | 2023b
#
################################################################################
"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from tqdm.auto import tqdm
from copy import deepcopy
from mlp_numpy import MLP
from modules import *
import matplotlib.pyplot as plt

import cifar10_utils

import torch




def evaluate_model(model, data_loader, num_classes=10):

    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
        accuracy 

    TODO:
    Implement evaluation of the MLP model on a given dataset.

    Hint: make sure to return the average accuracy of the whole dataset,
          independent of batch sizes (not all batches might be the same size).
    """
    correct = 0
    total = 0
    for inputs, labels in data_loader:
      inputs = inputs.reshape((inputs.shape)[0],-1)#changing the input dim to be s*n
      #print("inputs_validation",inputs.shape)
      # Forward pass
      outputs = model.forward(inputs)
        
      # Compute predicted labels
      predicted = np.argmax(outputs, axis=1)
        
      # Count correct predictions
      correct += np.sum(predicted == labels)
      total += labels.shape[0]

      # Compute validation accuracy
      accuracy = correct / total
      #print(f"Accuracy of the model: {100.0 * accuracy}%")


    return accuracy
  



def train(hidden_dims, lr, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that 
                     performed best on the validation. Between 0.0 and 1.0
      logging_info: An arbitrary object containing logging information. This is for you to 
                    decide what to put in here.

    TODO:
    - Implement the training of the MLP model. 
    - Evaluate your model on the whole validation set each epoch.
    - After finishing training, evaluate your model that performed best on the validation set, 
      on the whole test dataset.
    - Integrate _all_ input arguments of this function in your training. You are allowed to add
      additional input argument if you assign it a default value that represents the plain training
      (e.g. '..., new_param=False')

    Hint: you can save your best model by deepcopy-ing it.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    ## Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=True)

    
    subset = cifar10['train']
    subset_dataset = subset.dataset
    data, targets = subset_dataset[0]  # Access the first sample in the dataset
    data = data.numpy()

    input_dim = np.prod(data.shape)
    num_classes = len(subset_dataset.classes)
    n_hidden = args.hidden_dims
    # print("input_dim",input_dim,'\n',"n_hidden",n_hidden,'\n',"num_classes",num_classes)
    # print("type_input_dim",type(input_dim),'\n',"type_n_hidden",type(n_hidden),'\n',"type_num_classes",type(num_classes))

    # TODO: Initialize model and loss module
    model = MLP(input_dim, n_hidden, num_classes)
    loss_module =  CrossEntropyModule()

    # TODO: Training loop including validation

    val_accuracies = []
    training_losses = []
    best_model = None
    best_val_accuracy = 0.0
    for epoch in range(epochs):
        epoch_samples = 0
        epoch_loss = 0.0

        # Training
        
        for inputs, labels in cifar10_loader['train']:
            # Forward pass
            batch_loss = 0.0
            batch_samples = 0

            inputs = inputs.reshape((inputs.shape)[0],-1)#changing the input dim to be s*n
            outputs = model.forward(inputs)
            # Compute loss
            batch_loss = loss_module.forward(outputs, labels)
            batch_samples = inputs.shape[0]
            epoch_loss += (batch_loss *batch_samples)
            epoch_samples += batch_samples
            #print("loss",loss)
            ''''''''''''
            i, j= 2, 4
            eps = 0.00001
            model.layers[0].weight[i,j] += eps
            out = model.forward(inputs)
            loss1 = loss_module.forward(out, labels)     

            # approximate the gradient numerically 
            #print('Numerical gradient:', (loss1-loss)/eps) 
            ''''''''''''
            # Backward pass and update weights
            grad_loss = loss_module.backward(outputs, labels)
            model.backward(grad_loss)
            #print('Backprop  gradient:' , model.layers[0].grads['weight'][i,j])
            ##update here the weights and the biases with learing rate. 
            for layer in model.layers:
              if isinstance(layer,LinearModule):
                layer.weight -= lr * layer.grads['weight']
                layer.bias -= lr * layer.grads['bias']
            #model.clear_cache()
            #softmax_module.clear_cache()
        #print("Finish ephoc////////////////////////////////////////////////////")
            
        # Validation
         
        epoch_loss = epoch_loss /epoch_samples

        val_accuracy = evaluate_model(model, cifar10_loader['validation'], num_classes)
        print(f"Epoch {epoch + 1}/{epochs} - Validation Accuracy: {100.0 *val_accuracy:.4f}")
        print(f"Epoch {epoch + 1}/{epochs} - Training Loss: {epoch_loss:.4f}")

        #print("type",type(val_accuracy),val_accuracy)
        #print("type",type(best_val_accuracy),best_val_accuracy)

        val_accuracies.append(val_accuracy)
        training_losses.append(epoch_loss)

        # Check if current model has better validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = deepcopy(model)
    # TODO: Test best model
    test_accuracy = evaluate_model(best_model, cifar10_loader['test'], num_classes)
    # TODO: Add any information you might want to save for plotting
    logging_dict = {
        'input_dim': input_dim,
        'num_classes': num_classes,
        'best_val_accuracy': max(val_accuracies), 
        'training_losses':training_losses,
    }

    return best_model, val_accuracies, test_accuracy, logging_dict

def plot_3(dict_result, name):
    ephocs = 10
    for key in dict_result:
      plt.figure()  
      plt.plot(range(1, ephocs+1), dict_result[key])
      plt.xlabel('Epoch')
      plt.ylabel(key)
      plt.title(f'{key} per Epoch for {name}')
      plt.show() 

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    
    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    args = parser.parse_args()
    kwargs = vars(args)

    model, val_accuracies, test_accuracy, logging_dict = train(**kwargs)
    print(model, val_accuracies, test_accuracy, logging_dict)
    
    # plot(dict_result_numpy,"MLP_numpy")
    # Feel free to add any additional functions, such as plotting of the loss curve here
 

    