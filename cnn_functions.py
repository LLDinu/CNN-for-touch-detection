import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from tensorflow.python.framework import ops
from cnn_utils import *
from cnn_evaluation import *


def create_placeholders(n_Hx, n_Wx, n_Cx, n_Hy, n_Wy, n_Cy):
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_Hx -- scalar, height of the input image
    n_Wx -- scalar, width of the input image
    n_Cx -- scalar, number of input channels
    n_Hy -- scalar, height of the output image
    n_Wy -- scalar, width of the output image
    n_Cy -- scalar, number of output channels
        
    Returns:
    X -- placeholder for the input data, of shape [None, n_Hx, n_Wx, n_Cx] and dtype "float"
    Y -- placeholder for the input labels, of shape [None, n_Hy, n_Wy, n_Cy] and dtype "float"
    """

    X = tf.placeholder(tf.float32, shape = (None, n_Hx, n_Wx, n_Cx))
    Y = tf.placeholder(tf.float32, shape = (None, n_Hy, n_Wy, n_Cy))
    
    return X, Y


def initialize_parameters(n_Cx, n_Cy, n_C1 = 8, n_C2 = 16, n_C3 = 32):
    """
    Initializes weight parameters to build a neural network with tensorflow. The shapes are:
                        W1  : [3, 3, n_Cx, n_C1]
                        b1  : [1, 1,    1, n_C1]
                        W2a : [3, 3, n_C1,    1]
                        b2a : [1, 1,    1, n_C1]
                        W2b : [1, 1, n_C1, n_C2]
                        b2b : [1, 1,    1, n_C2]
                        W3a : [3, 3, n_C2,    1]
                        b3a : [1, 1,    1, n_C2]
                        W3b : [1, 1, n_C2, n_C3]
                        b3b : [1, 1,    1, n_C3]
                        W4  : [1, 1, n_C3, n_Cy]
                        b4  : [1, 1,    1, n_Cy]
                        
    Arguments:
    n_Cx -- scalar, number of input channels
    n_C1 -- scalar, number of 1st CONV layer filters
    n_C2 -- scalar, number of 2nd CONV layer filters
    n_C3 -- scalar, number of 3rd CONV layer filters
    n_Cy -- scalar, number of output channels
    Returns:
    parameters -- a dictionary of tensors containing the weights and biases (W1, b1, ..., W4 and b4)
    """
    
    W1 = tf.get_variable("W1", [3, 3, n_Cx, n_C1], initializer = tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("b1", [1, 1, 1, n_C1], initializer = tf.zeros_initializer())
    
    W2a = tf.get_variable("W2a", [3, 3, n_C1, 1], initializer = tf.contrib.layers.xavier_initializer())
    b2a = tf.get_variable("b2a", [1, 1, 1, n_C1], initializer = tf.zeros_initializer())
    W2b = tf.get_variable("W2b", [1, 1, n_C1, n_C2], initializer = tf.contrib.layers.xavier_initializer())
    b2b = tf.get_variable("b2b", [1, 1, 1, n_C2], initializer = tf.zeros_initializer())
    
    W3a = tf.get_variable("W3a", [3, 3, n_C2, 1], initializer = tf.contrib.layers.xavier_initializer())
    b3a = tf.get_variable("b3a", [1, 1, 1, n_C2], initializer = tf.zeros_initializer())
    W3b = tf.get_variable("W3b", [1, 1, n_C2, n_C3], initializer = tf.contrib.layers.xavier_initializer())
    b3b = tf.get_variable("b3b", [1, 1, 1, n_C3], initializer = tf.zeros_initializer())
    
    W4 = tf.get_variable("W4", [1, 1, n_C3, n_Cy], initializer = tf.contrib.layers.xavier_initializer())
    b4 = tf.get_variable("b4", [1, 1, 1, n_Cy], initializer = tf.zeros_initializer())
    
    parameters = {"W1": W1, "b1": b1,
                  "W2a": W2a, "b2a": b2a,
                  "W2b": W2b, "b2b": b2b,
                  "W3a": W3a, "b3a": b3a,
                  "W3b": W3b, "b3b": b3b,
                  "W4": W4, "b4": b4}
    
    return parameters


def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV_DW + CONV_PW -> RELU -> CONV_DW + CONV_PW -> RELU -> CONV2D
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing the parameters (weights and biases)
                  the shapes are given in initialize_parameters

    Returns:
    A1 -- the output of the 1st CONV layer
    A2 -- the output of the 2nd CONV layer
    A3 -- the output of the 3rd CONV layer
    Z4 -- the output of the last CONV layer
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    W2a = parameters['W2a']
    W2b = parameters['W2b']
    W3a = parameters['W3a']
    W3b = parameters['W3b']
    W4 = parameters['W4']
    b1 = parameters['b1']
    b2a = parameters['b2a']
    b2b = parameters['b2b']
    b3a = parameters['b3a']
    b3b = parameters['b3b']
    b4 = parameters['b4']

    # CONV2D: filters W1, stride 1, padding 'SAME'
    Z1 = tf.nn.conv2d(X, W1, strides = [1,1,1,1], padding = 'SAME')
    Z1 += b1 
    # RELU
    A1 = tf.nn.relu(Z1)
    # MAXPOOL: window 2x2, stride 2, padding 'SAME'
    P1 = tf.nn.max_pool(A1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
    
    # CONV_DW: filters W2a, stride 1, padding 'SAME'
    Z2a = tf.nn.depthwise_conv2d_native(P1, W2a, strides = [1,1,1,1], padding = 'SAME')
    Z2a += b2a
    # CONV_PW: filters W2b, stride 1, padding 'SAME'
    Z2b = tf.nn.conv2d(Z2a, W2b, strides = [1,1,1,1], padding = 'SAME')
    Z2b += b2b
    # RELU
    A2 = tf.nn.relu(Z2b)
    
    # CONV_DW: filters W3a, stride 1, padding 'SAME'
    Z3a = tf.nn.depthwise_conv2d_native(A2b, W3a, strides = [1,1,1,1], padding = 'SAME')
    Z3a += b3a
    # CONV_PW: filters W3b, stride 1, padding 'SAME'
    Z3b = tf.nn.conv2d(Z3a, W3b, strides = [1,1,1,1], padding = 'SAME')
    Z3b += b3b
    # RELU
    A3 = tf.nn.relu(Z3b)
    
    # CONV2D: filters W4, stride 1, padding 'SAME'
    Z4 = tf.nn.conv2d(A3b, W4, strides = [1,1,1,1], padding = 'SAME')
    Z4 += b4
    
    return A1, A2, A3, Z4


def compute_cost(Z4, Y, lambda_touch = 4, lambda_no_touch = 0.8, lambda_coord = 0.5, lambda_class = 0.5, gamma = 4.5):
    """
    Computes the cost function
    
    Arguments:
    Z4 -- output of forward propagation (output of the last CONV unit), of shape (number of examples, 8, 16, 5)
    Y -- "true" labels vector placeholder, same shape as Z4
    lambda_touch -- scalar, parameter which weighs the touch prediction loss where a touch is present 
    lambda_no_touch -- scalar, parameter which weighs the touch prediction loss where a touch is not present 
    lambda_coord -- scalar, parameter which weighs the coordinate regression loss
    lambda_class -- scalar, parameter which weighs the class prediction loss
    gamma -- scalar, focal loss parameter tuning the misclassification penalty
    
    Returns:
    cost -- tensor of the overall cost function
    cost_coord -- tensor of the coordinate cost function
    cost_touch_present -- tensor of the touch presence cost function
    """
    # Create a mask for evaluating cost only at labelled touch locations
    mask = tf.stack([Y[:,:,:,0], Y[:,:,:,0]], axis = -1)
    
    # Evaluate the individual cost function contributors
    cost_touch_present = focal_loss(Z4[:,:,:,0], Y[:,:,:,0], lambda_touch, lambda_no_touch, gamma)
    cost_coord = lambda_coord * tf.reduce_sum(tf.square(tf.multiply(tf.subtract(Y[:,:,:,1:3], Z4[:,:,:,1:3]), mask)))
    cost_class = lambda_class * tf.reduce_sum(tf.square(tf.multiply(tf.subtract(Y[:,:,:,3:5], Z4[:,:,:,3:5]), mask)))
    
    # Cost function
    cost  = cost_touch_present + cost_coord + cost_class
    
    return cost, cost_coord, cost_touch_present


def train_model(X_train, Y_train, X_test, Y_test, learning_rate,
          num_epochs, minibatch_size, print_cost = True):
    """
    Implements and trains a four-layer ConvNet in Tensorflow:
    CONV2D -> RELU -> MAXPOOL -> CONV_DW + CONV_PW -> RELU -> CONV_DW + CONV_PW -> RELU -> CONV2D
    
    Arguments:
    X_train -- training set data, of shape [None, n_Hx, n_Wx, n_Cx]
    Y_train -- training set labels, of shape [None, n_Hy, n_Wy, n_Cy]
    X_test -- test set data, of shape [None, n_Hx, n_Wx, n_Cx]
    Y_test -- test set labels, of shape [None, n_Hy, n_Wy, n_Cy]
    learning_rate -- learning rate of the optimizer
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- "True" to print the cost every 5 epochs
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    (m, n_Hx, n_Wx, n_Cx) = X_train.shape             # extract the input and output shapes
    (_, n_Hy, n_Wy, n_Cy) = Y_train.shape
    costs = []                                        # to keep track of the costs
    costs_coord = []
    costs_touch_present = []
    precisions_train = []                             # to keep track of the precision/recall on train/test sets
    precisions_test = []
    recalls_train = []
    recalls_test = []
    err_dists_train = []                              # to keep track of the coordinate distance error on train/test sets
    err_dists_test = []
    
    # Create Placeholders
    X, Y = create_placeholders(n_Hx, n_Wx, n_Cx, n_Hy, n_Wy, n_Cy)
    
    # Initialize parameters
    parameters = initialize_parameters(n_Cx, n_Cy)
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    _,_,_,Z4 = forward_propagation(X, parameters)
    
    # Cost function: Add cost function to tensorflow graph
    cost, cost_coord, cost_touch_present = compute_cost(Z4, Y)
    
    # Backpropagation: Define the tensorflow optimizer (AdamOptimizer) that minimizes the cost.
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    
    # Calculate the correct predictions (considering all elements of the output image)
    correct_prediction = tf.equal(tf.greater(Z4[:,:,:,0], 0.), tf.greater(Y[:,:,:,0], 0.5))
    
    # Calculate accuracy on the test set (considering all elements of the output image)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    
    # Initialize all the variables globally
    init = tf.global_variables_initializer()
    
    # Create a saver object which will save all the variables
    saver = tf.train.Saver()
    model_path = "C:/Users/OneMoreCookie/Jupyter_notes/model_v5/model.ckpt"
    
    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):

            minibatch_cost = 0.
            minibatch_cost_coord = 0.
            minibatch_cost_touch_present = 0.
            num_minibatches = int(m / minibatch_size) # compute number of minibatches of size minibatch_size in the train set
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size)

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                # Run the session to execute the optimizer and the cost
                _ , temp_cost, temp_cost_coord, temp_cost_touch_present = sess.run([optimizer, cost, cost_coord, cost_touch_present], 
                                                                                   feed_dict={X: minibatch_X, Y: minibatch_Y})
                
                minibatch_cost += temp_cost / num_minibatches
                minibatch_cost_coord += temp_cost_coord / num_minibatches 
                minibatch_cost_touch_present += temp_cost_touch_present / num_minibatches
            
            # Store the costs every epoch
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)
                costs_coord.append(minibatch_cost_coord)
                costs_touch_present.append(minibatch_cost_touch_present)
            
            # Print the cost and cost breakdown every 5 epoch
            if print_cost == True and epoch % 5 == 0:
                total_cost = minibatch_cost + 1e-8
                touch_cost = minibatch_cost_touch_present / total_cost
                coord_cost = minibatch_cost_coord / total_cost
                class_cost = (minibatch_cost - minibatch_cost_coord - minibatch_cost_touch_present) / total_cost
                
                print ("Cost after epoch " + str(epoch) + ": " + "{:.3f}".format(total_cost) + " (touch: " + "{:.2%}".format(touch_cost) + ", coord: " + "{:.2%}".format(coord_cost) + ", class: " + "{:.2%}".format(class_cost) + ")")
            
            # Evaluate the precision/recall of the train/test sets every 25 epochs
            if print_cost == True and epoch % 25 == 0 and epoch > 49:    
                z = Z4.eval({X: X_train})
                precision_train, recall_train, err_dist_train = test_model_during_training(z, Y_train)
                precisions_train.append(precision_train)
                recalls_train.append(recall_train)
                err_dists_train.append(err_dist_train)
                
                z = Z4.eval({X: X_test})
                precision_test, recall_test, err_dist_test = test_model_during_training(z, Y_test)
                precisions_test.append(precision_test)
                recalls_test.append(recall_test)
                err_dists_test.append(err_dist_test)
                
                print(" 1. Train set precision = " + str(precision_train))
                print(" 1. Train set recall = " + str(recall_train))
                print(" 1. Train set error distance = " + str(err_dist_train))
                print(" 2. Test set precision = " + str(precision_test))
                print(" 2. Test set recall = " + str(recall_test))
                print(" 2. Test set error distance = " + str(err_dist_test))  
        
        # Plot the cost
        if print_cost == True:
            plt.figure(1)
            plt.plot(np.squeeze(costs), 'b')
            plt.plot(np.squeeze(costs_coord), 'r')
            plt.plot(np.squeeze(costs_touch_present), 'g')
            plt.ylabel('cost')
            plt.xlabel('iterations')
            plt.title("Learning rate =" + str(learning_rate))

            plt.figure(2)
            plt.plot(100.-100.*np.squeeze(precisions_train), 'b')
            plt.plot(100.-100.*np.squeeze(precisions_test), 'r')
            plt.ylabel('precision loss(%)')
            plt.xlabel('iterations (per 25)')
            plt.title('train/test precision loss vs epochs')

            plt.figure(3)
            plt.plot(100.-100.*np.squeeze(recalls_train), 'b')
            plt.plot(100.-100.*np.squeeze(recalls_test), 'r')
            plt.ylabel('recall loss(%)')
            plt.xlabel('iterations (per 25)')
            plt.title('train/test recall loss vs epochs')
            plt.show()
        else:
            z = Z4.eval({X: X_train})
            precision_train, recall_train, err_dist_train = test_model_during_training_v5(z, Y_train)
            z = Z4.eval({X: X_test})
            precision_test, recall_test, err_dist_test = test_model_during_training_v5(z, Y_test)
            print(" 1. Train set precision = " + str(precision_train))
            print(" 1. Train set recall = " + str(recall_train))
            print(" 1. Train set error distance = " + str(err_dist_train))
            print(" 2. Test set precision = " + str(precision_test))
            print(" 2. Test set recall = " + str(recall_test))
            print(" 2. Test set error distance = " + str(err_dist_test)) 
        
        # Calculate the accuracy on the training set and the testing set (considering all elements of the output image)
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)
                
        #Save the graph
        save_path = saver.save(sess, model_path)
        print("Model saved in file: %s" % save_path)
        
        return parameters    
    
    
