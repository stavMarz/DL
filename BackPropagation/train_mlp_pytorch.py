################################################################################
#
# BGU IEM Introduction to Deep Learning Course | 2023b
#
################################################################################
"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from copy import deepcopy
from tqdm.auto import tqdm
from mlp_pytorch import MLP
import cifar10_utils
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import logging

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
    model.eval() #sets the model to evaluation mode
    correct, total_samples = 0., 0. #initialized to keep track of the number of correct predictions and the total number of samples processed

    with torch.no_grad():
        for data_inputs, data_labels in data_loader:
            batch_size = data_inputs.size(0) # s
            data_inputs = data_inputs.view(batch_size, -1) #flattens the input tensor
            outputs = model(data_inputs) #forward for the validation set
            probabilities = torch.softmax(outputs, dim=1) # Apply softmax to obtain class probabilities
            _, pred = torch.max(probabilities.data, 1) # predicted class index with the highest probability for each sample
            total_samples += data_labels.size(0) # add number of sample for each batch size
            #print("total_samples",total_samples) # for all the batch size
            correct += (pred == data_labels).sum().item() 

    accuracy = correct / total_samples
    #print(f"Accuracy of the model: {100.0 * accuracy}%")
    return accuracy


def train(hidden_dims, lr, use_batch_norm, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      use_batch_norm: If True, adds batch normalization layer into the network.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that 
                     performed best on the validation.
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
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False

    # Set default device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=False)

    subset = cifar10['train']
    subset_dataset = subset.dataset
    data, targets = subset_dataset[0]  # Access the first sample in the dataset

    input_dim = np.prod(data.shape)
    num_classes = len(subset_dataset.classes)

    learning_rates = np.logspace(-6, 2, num=9)  # 9 logarithmically spaced learning rates from 0.000001 to 100 for hyper parameters tuning
    scheduler_LR = []
    best_model = None
    # per ephoc seperatly 
    val_accuracies = [] 
    training_losses = []
   
    
    # for hyper parameters tuning 
    best_learningRates = []
    best_accuracies = []
    loss_curves = {}  # Dictionary to store loss curves for different learning rates
    for lr in learning_rates: #learning rate tuning loop 
        best_val_accuracy = 0.0
        model = MLP(input_dim, hidden_dims, num_classes).to(device) # model per lr 
        print("model",model)
        loss_module = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=lr)
        #scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[4,8], gamma=0.1) #4.b2
      
        for epoch in tqdm(range(epochs)):
            model.train()
            total_loss = 0.0
            total_samples = 0.0
            for inputs, labels in cifar10_loader['train']:
                inputs = inputs.to(device)
                labels = labels.to(device)
                inputs = inputs.view(inputs.size(0), -1)
                optimizer.zero_grad()
                outputs = model(inputs) #forward 
                loss = loss_module(outputs, labels)
                loss.backward()#backward
                optimizer.step()# Update weights
                total_loss += loss.item() * inputs.size(0) ##add the loss for each batch 
                total_samples += inputs.size(0)
                #lr = optimizer.param_groups[0]['lr']
            epoch_loss = np.nan_to_num(total_loss) / total_samples ## calculate the total loss for the ephoc 
            training_losses.append(epoch_loss)
            logging.info(f"Epoch {epoch + 1}/{epochs} - Training Loss: {epoch_loss:.4f}")
            print(f"Epoch {epoch + 1}/{epochs} - Training Loss: {epoch_loss:.4f}")
            print(f"Epoch {epoch + 1}: Learning Rate = {optimizer.param_groups[0]['lr']}")

            model.eval() ##evaluation for each ephoc 
            val_accuracy = evaluate_model(model, cifar10_loader['validation'], device)
            val_accuracies.append(val_accuracy)
            #scheduler_LR.append(scheduler.get_last_lr()) 
            #scheduler.step() #4.b2

            logging.info(f"Epoch {epoch + 1}/{epochs} - Validation Accuracy: {val_accuracy:.4f}%")
            print(f"Epoch {epoch + 1}/{epochs} - Validation Accuracy: {100.0 *val_accuracy:.4f}%")

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model = deepcopy(model)

                    
            if lr not in loss_curves:
                loss_curves[lr] = [epoch_loss]
            else:
                loss_curves[lr].append(epoch_loss)
            
        
        best_accuracies.append(best_val_accuracy)

    print("best_accuracies",len(best_accuracies),best_accuracies)
        
    test_accuracy = evaluate_model(best_model, cifar10_loader['test'], device)
    logging.info(f"Best Validation Accuracy: {best_val_accuracy:.4f}")
    logging.info(f"Test Accuracy: {test_accuracy:.4f}")
    
    logging_info = {
      'input_dim': input_dim,
      'num_classes': num_classes,
      'device': device,
      'best_val_accuracy': max(val_accuracies), 
      'training_losses':training_losses,
      #'scheduler_LR':scheduler_LR,
      'learning_Rate':learning_rates,
      'best_accuracies':best_accuracies,
      'loss_curves':loss_curves
    }

    
    # Print the best learning rate and its corresponding accuracy
    best_learning_rate = learning_rates[np.argmax(best_accuracies)]
    print(f"The best learning rate is {best_learning_rate} with a validation accuracy of {max(val_accuracies):.4f}")
    print(f"Test Accuracy with the best model: {test_accuracy:.4f}")
    
    return best_model, val_accuracies, test_accuracy, logging_info


def plot_4a(dict_result, name):
    ephocs = 10
    for key in dict_result:
      plt.figure()  
      plt.plot(range(1, ephocs+1), dict_result[key])
      plt.xlabel('Epoch')
      plt.ylabel(key)
      plt.title(f'{key} per Epoch for {name}')
      plt.show()
    
def plot_4bi(dict_result, name):
  ephocs = 10
  for key in dict_result:
    plt.figure()  
    plt.plot(range(1, ephocs+1), dict_result[key])
    plt.xlabel('Epoch')
    plt.ylabel(key)
    plt.title(f'{key} per Epoch for {name}')
    plt.show()

def plot_4biv(accuracies,learningRates):
  plt.figure()  
  plt.plot(accuracies,learningRates, marker='o')
  plt.xlabel('learning rate')
  plt.ylabel('Accuracy')
  plt.title("Best accuracy per learning rate")
  plt.show()

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    parser.add_argument('--use_batch_norm', action='store_true',
                        help='Use this option to add Batch Normalization layers to the MLP.')
    
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

    best_model, val_accuracies, test_accuracy, logging_info = train(**kwargs)
    print(best_model, val_accuracies, test_accuracy, logging_info)

    #dict_result_pytorch = {"Accuracy":val_accuracies,"training losses":logging_info['training_losses'],'learning rates':logging_info['learning_Rate']}
    #plot_4a(dict_result_pytorch,"MLP_pytorch")
    #plot_4bi(learning_dict,"with scheduler") 
    #plot_4biv(learning_dict['learning_Rate','best_accuracies'])

    # Feel free to add any additional functions, such as plotting of the loss curve here
    