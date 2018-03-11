import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops

def load_dataset():
    """
    Loads the labelled data and splits it into train/test datasets
    
    Returns:
    train_set_x -- train input data
    train_set_y -- train data labels
    test_set_x -- test input data
    test_set_y -- test data labels
    """    
    
    # Load inputs and labels corresponding to class 1 (grounded touch) 
    dataset = h5py.File('gnd_data_v4.h5', "r")
    
    x0 = np.array(dataset["x_self_data"][:])    # train set features (reshaped and normalized)
    x0 = x0.T
    x0 = x0.reshape(x0.shape[0],16,32)
    x0 = -x0/1000.                              
    x1 = np.array(dataset["x_mut_data"][:])     # train set features (reshaped and normalized)
    x1 = x1.T
    x1 = x1.reshape(x1.shape[0],16,32)
    x1 = -x1/1000.
    x_data_1 = np.stack((x0, x1), axis = 3)     # stack the input data (2 input channels)
    
    y0 = np.array(dataset["touch_present"][:])  # train set labels
    y0 = y0.T
    y0 = y0.reshape(y0.shape[0],8,16)
    y1 = np.array(dataset["x_coord"][:])        # train set labels
    y1 = y1.T
    y1 = y1.reshape(y1.shape[0],8,16)    
    y2 = np.array(dataset["y_coord"][:])        # train set labels
    y2 = y2.T
    y2 = y2.reshape(y2.shape[0],8,16)           # train set class labels
    y3 = y0
    y4 = y0*0.
    y_data_1 = np.stack((y0, y1, y2, y3, y4), axis = 3) # stack the output data (5 output channels)
    
    # Load inputs and labels corresponding to class 2 (floating touch)
    dataset = h5py.File('float_data_v4.h5', "r")
    
    x0 = np.array(dataset["x_self_data"][:]) # train set features (reshaped and normalized)
    x0 = x0.T
    x0 = x0.reshape(x0.shape[0],16,32)
    x0 = -x0/1000.
    x1 = np.array(dataset["x_mut_data"][:]) # train set features (reshaped and normalized)
    x1 = x1.T
    x1 = x1.reshape(x1.shape[0],16,32)
    x1 = -x1/1000.
    x_data_2 = np.stack((x0, x1), axis = 3)     # stack the input data (2 input channels)
    
    y0 = np.array(dataset["touch_present"][:])  # train set labels
    y0 = y0.T
    y0 = y0.reshape(y0.shape[0],8,16)
    y1 = np.array(dataset["x_coord"][:])        # train set labels
    y1 = y1.T
    y1 = y1.reshape(y1.shape[0],8,16)    
    y2 = np.array(dataset["y_coord"][:])        # train set labels
    y2 = y2.T
    y2 = y2.reshape(y2.shape[0],8,16)           # train set class labels
    y3 = y0*0.
    y4 = y0
    y_data_2 = np.stack((y0, y1, y2, y3, y4), axis = 3) # stack the output data (5 output channels)
    
    # Merge the inputs and labels for all classes
    x_data = np.concatenate([x_data_1, x_data_2], axis = 0)
    y_data = np.concatenate([y_data_1, y_data_2], axis = 0)
    
    # Shuffle and split the inputs and labels into training (80%) and testing (20%) data sets
    np.random.seed(1)                          # constant seed for reproducing the results
    m = x_data.shape[0]
    perm = np.random.permutation(range(m))
    shuffled_x_data = x_data[perm,:,:,:]
    shuffled_y_data = y_data[perm,:,:,:]
    
    m_partition = math.floor(m*0.8)
    train_set_x = shuffled_x_data[0:m_partition,:,:,:]
    train_set_y = shuffled_y_data[0:m_partition,:,:,:]
    test_set_x = shuffled_x_data[m_partition::,:,:,:]
    test_set_y = shuffled_y_data[m_partition::,:,:,:]
    
    return train_set_x, train_set_y, test_set_x, test_set_y

def random_mini_batches(X, Y, mini_batch_size):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    
    # Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]

    # Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def focal_loss(prediction_tensor, target_tensor, alpha_obj, alpha_noobj, gamma):
    """
    Compute focal loss for predictions.
    Multi-labels Focal loss formula:
            FL = -alpha_obj * (z-p)^gamma * log(p) - alpha_noobj * p^gamma * log(1-p)
                 ,where alpha = TBD, gamma = 2, p = sigmoid(x), z = target_tensor.
                 
    Arguments:
     prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     # weights: A float tensor of shape [batch_size, num_anchors]
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
     
    Returns:
        loss: a tensor representing the value of the loss function
    """
    sigmoid_p = tf.nn.sigmoid(prediction_tensor)
    # sigmoid_p = prediction_tensor
    cost_touch_present = tf.multiply(target_tensor, 
                                     tf.multiply(tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)), 
                                                 tf.pow(1 - sigmoid_p, gamma)))
    cost_touch_not_present = tf.multiply(1 - target_tensor, 
                                         tf.multiply(tf.log(tf.clip_by_value(1 - sigmoid_p, 1e-8, 1.0)), 
                                                     tf.pow(sigmoid_p, gamma)))
    per_entry_cross_ent = - alpha_obj * cost_touch_present - alpha_noobj * cost_touch_not_present
    
    return tf.reduce_sum(per_entry_cross_ent)

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y
