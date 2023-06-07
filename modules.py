################################################################################
#
# BGU IEM Introduction to Deep Learning Course | 2023b
#
################################################################################

"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np


class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """

    def __init__(self, in_features, out_features, input_layer=False):
        """
        Initializes the parameters of the module.

        Args:
          in_features: size of each input sample
          out_features: size of each output sample
          input_layer: boolean, True if this is the first layer after the input, else False.

        TODO:
        Initialize weight parameters using Kaiming initialization. 
        Initialize biases with zeros.
        Hint: the input_layer argument might be needed for the initialization

        Also, initialize gradients with zeros.
        """

        self.in_features = in_features
        self.out_features = out_features
        self.input_layer = input_layer
        #in_features_len = len(list(in_features))
        # Initialize weights using Kaiming initialization - 
        
        std = np.sqrt(2 / in_features)  # For all the layer because the factor 1/2 does not matter if it just exists on the firstone layer.for simplicity.
        
        #print("in_features",in_features)
        #print("out_features",out_features)

        self.weight = np.random.randn(out_features,in_features)* std
        #print(self.weight.shape)
        # Initialize biases with zeros
        self.bias = np.zeros((1, out_features)) ##to check if needed n*m
        # Initialize gradients with zeros
        self.grads = {'weight': np.zeros_like(self.weight), 'bias': np.zeros_like(self.bias)}

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        self.x = x
        #print("x_shape",x.shape)
        #print("wt_shape",self.weight.transpose().shape)
        self.bias = np.zeros((self.x.shape[0], self.out_features))
        #print("b_shape",self.bias.shape)
        #print(np.dot(self.x,self.weight.transpose()))
        y= (np.dot(self.x,self.weight.transpose()))+self.bias
        #print("y_shape",y.shape)

        #print("sucsess")
        # Compute the linear transformation
        out = y
        #out =  np.matrix(self.weight.transpose()) * np.matrix(x) + self.bias
        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to
        layer parameters in self.grads['weight'] and self.grads['bias'].
        """
        self.dout = dout
        #print("dout_shape",dout.shape)
        #print("x_shape",self.x.shape)

        # Compute gradients with respect to weights and biases - according to 2.a.1 and 2.a.2
        self.grads['weight'] = np.dot(self.dout.transpose(), self.x)
        self.grads['bias'] = np.sum(self.dout, axis=1, keepdims=True)

        # Compute gradients with respect to the input
        dx = np.dot(dout, self.weight)
        #print("dx_back_linear",dx.shape)
        
        if self.input_layer == False: 
          return dx
        else:
          return None

    def clear_cache(self):
        """
        Remove any saved numpy for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        self.x = None 


class ReluModule(object):
    """
    RELU activation module.
    """
    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.
        
        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        
       # Apply ReLU activation function
        out = np.maximum(0, x)

        # Store the input for later use in the backward pass
        self.input = x

        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        dx = dout * (self.input > 0)
        return dx

    def clear_cache(self):
        """
        Remove any saved numpy for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        self.input = None


class SoftMaxModule(object):
    """
    Softmax activation module.
    """

    def forward(self, x): #the defenition of softmax for the input
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.
        To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
     
        # Apply the Max Trick to stabilize computation
        shifted_x = x - np.max(x, axis=1, keepdims=True)
        
        # Compute the exponential of the shifted input
        exp_x = np.exp(shifted_x) #numerator 
        
        # Compute the sum of the exponential values along the second axis
        sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)#denominator
        
        # Compute the softmax output by dividing each exponential value by the sum
        out = exp_x / sum_exp_x

        # Store the softmax output for later use in the backward pass
        self.softmax_output = out
        #print("out",out.shape)
        return out

    def backward(self, dout):## according to 2.c.i
        """
        Backward pass.
        Args:
          dout: gradients of the previous modul ##grad_loss
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        N = self.softmax_output.shape[0]

        # Calculate the dot product between dout and the transposed softmax output
        dot_product = np.dot(dout, self.softmax_output.transpose())

        # Create an identity matrix with the same shape as the softmax output
        identity_matrix = np.eye(N)

        # Subtract the diagonal elements from the dot product result
        subtraction = dot_product*identity_matrix

        # Calculate the row-wise sums of the subtraction result
        ones = np.ones_like(self.softmax_output)
        row_sums = np.dot(subtraction,ones)

        # Calculate the final gradient
        dx = self.softmax_output * (dout - row_sums)
        
        
        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        self.softmax_output = None


class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """

    def forward(self, x, y):
        """
        Forward pass.
        Args:
          x: input to the module - from the softmax! 
          y: labels of the input
        Returns:
          out: cross entropy loss

        TODO:
        Implement forward pass of the module.
        """
  

        # num_samples = x.shape[0]
        # out = -np.sum(np.log(x[np.arange(num_samples), y])) / num_samples
        # Convert y to one-hot encoded vector
        num_classes = x.shape[1] # C
        #print("num_classes",num_classes)
        num_samples = x.shape[0] # S
        #print("num_samples",num_samples)
        epsilon = 1e-10  # Small constant to avoid division by zero
        y_onehot = np.zeros((len(y), num_classes))
        y_onehot[np.arange(len(y)), y] = 1
        #print("y_onehot",y_onehot.shape)
        # Compute the cross entropy loss
        loss_per_sample = -np.sum(y_onehot * np.log(x+epsilon), axis=1)
        total_loss = np.sum(loss_per_sample)
        out = total_loss / num_samples
        return out 
        
      ##### TO CHECK!!!!!!!!!!!!!!!!!!!!!
    def backward(self, x, y):## according to 2.c.11
        """
        Backward pass.
        Args:
          x: input to the module
          y: labels of the input - t
        Returns:
          dx: gradient of the loss with the respect to the input x.

        TODO:
        Implement backward pass of the module.
        """
        epsilon = 1e-10  # Small constant to avoid division by zero

        # Create a one-hot encoded version of y
        num_classes = x.shape[1] # C
        num_samples = x.shape[0] # S
        #print("x_back",x.shape)
        #dx = (-1/num_samples)*(y_onehot /x) # Compute the gradient of the loss with respect to x
        # Convert y to one-hot encoding
        y_onehot = np.zeros((num_samples,num_classes)) #matrix T
        y_onehot[np.arange(num_samples), y] = 1
        #print("y_back",y_onehot.shape)
        # Compute the gradient of the loss with respect to x
        dx = -(1 / num_samples) * np.divide(y_onehot,x+epsilon)
        #print("i have dx",dx.shape)
        
        return dx