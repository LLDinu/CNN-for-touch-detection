import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from tensorflow.python.framework import ops
from sympy.utilities.iterables import multiset_permutations
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from cnn_utils import *


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

def initialize_parameters(n_Cx, n_C1 = 8, n_C2 = 16, n_C3 = 32, n_Cy):
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
    mask = Y[:,:,:,0]
    mask2 = tf.stack([Y[:,:,:,0], Y[:,:,:,0]], axis = -1)
    
    cost_touch_present = focal_loss(Z4[:,:,:,0], Y[:,:,:,0], lambda_touch, lambda_no_touch, gamma)
    
    cost_coord = lambda_coord * tf.reduce_sum(tf.square(tf.multiply(tf.subtract(Y[:,:,:,1:3], Z4[:,:,:,1:3]), mask2)))
    
    cost_class = lambda_class * tf.reduce_sum(tf.square(tf.multiply(tf.subtract(Y[:,:,:,3:5], Z4[:,:,:,3:5]), mask2)))
    
    cost  = cost_touch_present + cost_coord + cost_class
    
    return cost, cost_coord, cost_touch_present

def train_model(X_train, Y_train, X_test, Y_test, learning_rate,
          num_epochs, minibatch_size, print_cost = True):
    """
    Implements and trains a four-layer ConvNet in Tensorflow:
    CONV2D -> RELU -> MAXPOOL -> CONV_DW + CONV_PW -> RELU -> CONV_DW + CONV_PW -> RELU -> CONV2D
    
    Arguments:
    X_train -- training set data, of shape (None, 16, 32, 2)
    Y_train -- training set labels, of shape (None, 8, 16, 4)
    X_test -- test set data, of shape (None, 16, 32, 2)
    Y_test -- test set labels, of shape (None, 8, 16, 4)
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
    
def run_model(X_data):
    """
    Runs the saved model on a given input data. Creates an excel report with formatted outputs, activations and parameters.
    
    Arguments:
    X_data -- input data, of shape [1, n_Hx, n_Wx, n_Cx]
    """
    
    save_path = 'D:/Work/Touch data/touch_data_python_results/processed_frame_v5.xlsx'
    workbook = pd.ExcelWriter(save_path, engine='xlsxwriter')
    
    # Load the weights
    w = h5py.File('weights_v4.h5', 'r')
    W1 = np.array(w['W1'][:])
    b1 = np.array(w['b1'][:])
    W2a = np.array(w['W2a'][:])
    b2a = np.array(w['b2a'][:])
    W2b = np.array(w['W2b'][:])
    b2b = np.array(w['b2b'][:])
    W3a = np.array(w['W3a'][:])
    b3a = np.array(w['b3a'][:])
    W3b = np.array(w['W3b'][:])
    b3b = np.array(w['b3b'][:])
    W4 = np.array(w['W4'][:])
    b4 = np.array(w['b4'][:])
    w.close()
    
    # Get data dimensions
    (m, n_Hx, n_Wx, n_Cx) = X_data.shape
    (n_Hy, n_Wy) = np.floor([n_Hx/2, n_Wx/2]).astype(int)

    # Create placeholder
    X = tf.placeholder(tf.float32, shape = (None, n_Hx, n_Wx, n_Cx))

    # Initialize parameters
    parameters = {"W1": W1, "b1": b1,
                  "W2a": W2a, "b2a": b2a,
                  "W2b": W2b, "b2b": b2b,
                  "W3a": W3a, "b3a": b3a,
                  "W3b": W3b, "b3b": b3b,
                  "W4": W4, "b4": b4}
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    A1, A2, A3, Z4 = forward_propagation(X, parameters)
    
    # Initialize all the variables globally
    init = tf.global_variables_initializer()
    
    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)

        # Evaluate the outputs of each layer
        z = Z4.eval({X: X_data})
        a1 = A1.eval({X: X_data})
        a2 = A2.eval({X: X_data})
        a3 = A3.eval({X: X_data})
        
        # Set the score and distance thresholds for non-maximum suppression 
        # (e.g. consider touches only if output probability > 50% and distance > 1.8 between touches)
        score_th = 50
        dist_th = 1.8
        
        # Store input data
        data0 = np.reshape(np.round(1000*X_data[0,:,:,0]), (n_Hx, n_Wx))
        data1 = np.reshape(np.round(1000*X_data[0,:,:,1]), (n_Hx, n_Wx))
        
        # Store touch presence score
        data2 = np.reshape(np.round(100/(1 + np.exp(-z[0,:,:,0]))), (n_Hy, n_Wy))
        
        # Store predicted X coordinates (expand normalized values)
        data3 = 2 * (np.reshape(z[0,:,:,1], (n_Hy, n_Wy)) + np.array(range(n_Wy))) - 0.5
        data3[np.where(data2 < score_th)] = 0
        
        # Store predicted Y coordinates (expand normalized values)
        data4 = 2 * (np.reshape(z[0,:,:,2], (n_Hy, n_Wy)) + np.reshape(np.array(range(n_Hy)), (n_Hy,1))) - 0.5
        data4[np.where(data2 < score_th)] = 0
        
        # Store class 1 (grounded touch) score
        data5 = np.reshape(np.round(100*z[0,:,:,3]), (n_Hy, n_Wy))
        data5[np.where(data2 < score_th)] = 0
        
        # Store class 2 (floating touch) score
        data6 = np.reshape(np.round(100*z[0,:,:,4]), (n_Hy, n_Wy))
        data6[np.where(data2 < score_th)] = 0
        
        # Store predicted touches
        touches = non_max_suppression(data2, data3, data4, score_th, dist_th)
        
        # Export above values to the worksheet
        wb = workbook.book
        format = wb.add_format({'align': 'center', 'valign': 'vcenter'}) 
        cell_format = wb.add_format({'align': 'center', 'valign': 'vcenter', 'bold': True, 'font_size': 12})
        df0 = pd.DataFrame(data0)
        df1 = pd.DataFrame(data1)
        df2 = pd.DataFrame(data2)
        df3 = pd.DataFrame(data3)
        df4 = pd.DataFrame(data4)
        df5 = pd.DataFrame(data5)
        df6 = pd.DataFrame(data6)
        dft = pd.DataFrame(touches)
        
        df0.to_excel(workbook, index=True, sheet_name='Outputs', startcol=0, startrow=42)
        df1.to_excel(workbook, index=True, sheet_name='Outputs', startcol=0, startrow=1)
        df2.to_excel(workbook, index=True, sheet_name='Outputs', startcol=3, startrow=20)
        df3.to_excel(workbook, index=True, sheet_name='Outputs', startcol=0, startrow=31)
        df4.to_excel(workbook, index=True, sheet_name='Outputs', startcol=17, startrow=31)
        dft.to_excel(workbook, index=True, sheet_name='Outputs', startcol=23, startrow=22)
        df5.to_excel(workbook, index=True, sheet_name='Outputs', startcol=0, startrow=61)
        df6.to_excel(workbook, index=True, sheet_name='Outputs', startcol=17, startrow=61)
        
        worksheet = workbook.sheets['Outputs']
        worksheet.conditional_format(2, 1, 17, 32, {'type': '2_color_scale',
                                                    'min_color': "#FFFFFF",
                                                    'max_color': "#63BE7B"})
        worksheet.conditional_format(21, 4, 28, 19, {'type': '2_color_scale',
                                                     'min_color': "#FFFFFF",
                                                     'max_color': "#63BE7B"})
        worksheet.conditional_format('B33:Q40', {'type': '2_color_scale',
                                                 'min_color': "#FFFFFF",
                                                 'max_color': "#63BE7B"})
        worksheet.conditional_format('S33:AH40', {'type': '2_color_scale',
                                                  'min_color': "#FFFFFF",
                                                  'max_color': "#63BE7B"})
        worksheet.conditional_format(43, 1, 58, 32, {'type': '2_color_scale',
                                                     'min_color': "#FFFFFF",
                                                     'max_color': "#63BE7B"})
        worksheet.conditional_format('B63:Q70', {'type': '2_color_scale',
                                                 'min_color': "#FFFFFF",
                                                 'max_color': "#63BE7B"})
        worksheet.conditional_format('S63:AH70', {'type': '2_color_scale',
                                                  'min_color': "#FFFFFF",
                                                  'max_color': "#63BE7B"})
        worksheet.set_column(0, 33, 4.3, format)
        worksheet.merge_range(0, 0, 0, 4, 'Input mutual raw data:', cell_format)
        worksheet.merge_range(41, 0, 41, 4, 'Input self raw data:', cell_format)
        worksheet.merge_range(19, 3, 19, 6, 'Touch presence:', cell_format)
        worksheet.merge_range(30, 0, 30, 3, 'X coordinates:', cell_format)
        worksheet.merge_range(30, 17, 30, 20, 'Y coordinates:', cell_format)
        worksheet.merge_range(21, 23, 21, 26, 'Detected touches:', cell_format)
        worksheet.merge_range(60, 0, 60, 3, 'Grounded touch:', cell_format)
        worksheet.merge_range(60, 17, 60, 20, 'Floating touch:', cell_format)
       
        # Save the activations A1, A2, A3 for visualization
        for i in range(a1.shape[3]):
            data = np.reshape(np.round(1000*a1[0,:,:,i]), (16,32))
            df = pd.DataFrame(data)
            df.to_excel(workbook, index=True, sheet_name='A1', startcol=0, startrow=1+i*19)
            worksheet = workbook.sheets['A1']
            worksheet.conditional_format(i*19+2, 1, i*19+17, 33, {'type': '2_color_scale',
                                                                  'min_color': "#FFFFFF",
                                                                  'max_color': "#63BE7B"})
            worksheet.set_column(0, 32, 4.3, format)
            worksheet.merge_range(i*19, 0, i*19, 3, 'A1_'+str(i+1)+':', cell_format)
        for i in range(a2.shape[3]):
            data = np.reshape(np.round(1000*a2[0,:,:,i]), (8,16))
            df = pd.DataFrame(data)
            df.to_excel(workbook, index=True, sheet_name='A2', startcol=(i%2)*17, startrow=1+math.floor(i/2)*11)
            worksheet = workbook.sheets['A2']
            worksheet.conditional_format(math.floor(i/2)*11+2, (i%2)*17+1, math.floor(i/2)*11+10, (i%2)*17+16, 
                                         {'type': '2_color_scale',
                                          'min_color': "#FFFFFF",
                                          'max_color': "#63BE7B"})
            worksheet.set_column(0, 33, 4.3, format)
            worksheet.merge_range(math.floor(i/2)*11, (i%2)*17, math.floor(i/2)*11, (i%2)*17+3, 'A2_'+str(i+1)+':', cell_format)
        for i in range(a3.shape[3]):
            data = np.reshape(np.round(1000*a3[0,:,:,i]), (8,16))
            df = pd.DataFrame(data)
            df.to_excel(workbook, index=True, sheet_name='A3', startcol=(i%2)*17, startrow=1+math.floor(i/2)*11)
            worksheet = workbook.sheets['A3']
            worksheet.conditional_format(math.floor(i/2)*11+2, (i%2)*17+1, math.floor(i/2)*11+10, (i%2)*17+16, 
                                         {'type': '2_color_scale',
                                          'min_color': "#FFFFFF",
                                          'max_color': "#63BE7B"})
            worksheet.set_column(0, 33, 4.3, format)
            worksheet.merge_range(math.floor(i/2)*11, (i%2)*17, math.floor(i/2)*11, (i%2)*17+3, 'A3_'+str(i+1)+':', cell_format)
            
        # Save the weights (W1-W4) and biases (b1-b4) for visualization
        for j in range(W1.shape[3]):
            for i in range(W1.shape[2]):
                data = np.reshape(W1[:,:,i,j], (3,3))
                df = pd.DataFrame(data)
                df.to_excel(workbook, index=True, sheet_name='W1', startcol=i*(3+1), startrow=1+j*8)
                worksheet = workbook.sheets['W1']
                worksheet.conditional_format(2+j*8, i*4+1, 4+j*8, i*4+3, {'type': '2_color_scale',
                                                                            'min_color': "#FFFFFF",
                                                                            'max_color': "#63BE7B"})
            worksheet.set_column(0, 8, 5, format)
            worksheet.merge_range(j*8, 0, j*8, 3, 'W1_'+str(j+1)+':', cell_format)
            worksheet.merge_range(5+j*8, 0, 5+j*8, 3, 'b1_'+str(j+1)+':', cell_format)
            worksheet.merge_range(6+j*8, 0, 6+j*8, 3, b1[0,0,0,j], format)    

        for i in range(W2a.shape[2]):
            data = np.reshape(W2a[:,:,i,0], (3,3))
            df = pd.DataFrame(data)
            df.to_excel(workbook, index=True, sheet_name='W2', startcol=i*(3+1), startrow=1)
            worksheet = workbook.sheets['W2']
            worksheet.conditional_format(2, i*4+1, 4, i*4+3, {'type': '2_color_scale',
                                                              'min_color': "#FFFFFF",
                                                              'max_color': "#63BE7B"})
            worksheet.merge_range(5, 0, 5, 3, 'b2a:', cell_format)
            worksheet.merge_range(6, 0+i*4, 6, 3+i*4, b2a[0,0,0,i], format)
        worksheet.set_column(0, 33, 5, format)
        worksheet.merge_range(0, 0, 0, 3, 'W2a:', cell_format)
            
        for i in range(W2b.shape[3]):
            data = np.reshape(W2b[0,0,:,i], (1,8))
            df = pd.DataFrame(data)
            df.to_excel(workbook, index=True, sheet_name='W2', startcol=0, startrow=9+i*6)
            worksheet = workbook.sheets['W2']
            worksheet.conditional_format(10+i*6, 1, 10+i*6, 16, {'type': '2_color_scale',
                                                                 'min_color': "#FFFFFF",
                                                                 'max_color': "#63BE7B"})
            worksheet.merge_range(8+i*6, 0, 8+i*6, 3, 'W2b_'+str(i+1)+':', cell_format)
            worksheet.merge_range(11+i*6, 0, 11+i*6, 3, 'b2b_'+str(i+1)+':', cell_format)
            worksheet.merge_range(12+i*6, 0, 12+i*6, 3, b2b[0,0,0,i], format)     
        
        for i in range(W3a.shape[2]):
            data = np.reshape(W3a[:,:,i,0], (3,3))
            df = pd.DataFrame(data)
            df.to_excel(workbook, index=True, sheet_name='W3', startcol=i*(3+1), startrow=1)
            worksheet = workbook.sheets['W3']
            worksheet.conditional_format(2, i*4+1, 4, i*4+3, {'type': '2_color_scale',
                                                              'min_color': "#FFFFFF",
                                                              'max_color': "#63BE7B"})
            worksheet.merge_range(5, 0, 5, 3, 'b3a:', cell_format)
            worksheet.merge_range(6, 0+i*4, 6, 3+i*4, b3a[0,0,0,i], format)
        worksheet.set_column(0, 65, 5, format)
        worksheet.merge_range(0, 0, 0, 3, 'W3a:', cell_format)
        
        for i in range(W3b.shape[3]):
            data = np.reshape(W3b[0,0,:,i], (1,16))
            df = pd.DataFrame(data)
            df.to_excel(workbook, index=True, sheet_name='W3', startcol=0, startrow=9+i*6)
            worksheet = workbook.sheets['W3']
            worksheet.conditional_format(10+i*6, 1, 10+i*6, 32, {'type': '2_color_scale',
                                                                 'min_color': "#FFFFFF",
                                                                 'max_color': "#63BE7B"})
            worksheet.merge_range(8+i*6, 0, 8+i*6, 3, 'W3b_'+str(i+1)+':', cell_format)
            worksheet.merge_range(11+i*6, 0, 11+i*6, 3, 'b3b_'+str(i+1)+':', cell_format)
            worksheet.merge_range(12+i*6, 0, 12+i*6, 3, b3b[0,0,0,i], format)  
        
        for i in range(W4.shape[3]):
            data = np.reshape(W4[0,0,:,i], (1,32))
            df = pd.DataFrame(data)
            df.to_excel(workbook, index=True, sheet_name='W4', startcol=0, startrow=1+i*6)
            worksheet = workbook.sheets['W4']
            worksheet.conditional_format(2+i*6, 1, 2+i*6, 32, {'type': '2_color_scale',
                                                               'min_color': "#FFFFFF",
                                                               'max_color': "#63BE7B"})
            worksheet.set_column(0, 33, 5, format)
            worksheet.merge_range(i*6, 0, i*6, 3, 'W4_'+str(i+1)+':', cell_format)
            worksheet.merge_range(3+i*6, 0, 3+i*6, 3, 'b4_'+str(i+1)+':', cell_format)
            worksheet.merge_range(4+i*6, 0, 4+i*6, 3, b4[0,0,0,i], format) 
        
        workbook.save()
        workbook.close()
    
def non_max_suppression(scores, x_coord, y_coord, score_th, distance_th):
    """
    Non-maximum suppression algorithm for eliminating redundant predictions.
    
    Arguments:
    scores -- array of scores, of shape (n_Hy, n_Wy)
    x_coord -- array of X coordinates, of shape (n_Hy, n_Wy)
    y_coord -- array of Y coordinates, of shape (n_Hy, n_Wy)
    score_th -- score threshold: eliminate elements (scores, x_coord, y_coord) whose score is < score_th
    distance_th -- distance threshold: eliminate elements within distance_th of higher probability elements 
    
    Returns:
    touches -- array of detected touches, of shape (3, number of detected touches)
    """
    
    # Flatten the matrices and stack them into a new matrix (3,-)
    scores = scores.flatten()
    x_coord = x_coord.flatten()
    y_coord = y_coord.flatten()
    data = np.stack((scores, x_coord, y_coord), axis=0)
    
    # Remove scores lower than "score_th"
    ind = np.argwhere(data[0,:] < score_th)
    data = np.delete(data, ind, axis=1)
    
    # Sort the remaining scores (flip the indices for descending order !)
    ind = data.argsort(axis=1)
    data = data[:,ind[0,::-1]]
    
    # Select and store the max (point with the highest score)
    if data.size > 0:
        touches = np.reshape(data[:,0],(3,1))
    else:
        touches = np.empty([3, 1])

    # Loop the steps: {select max} and {remove small distance elements}
    while data.size > 0:
        
        # Calculate the distances from coord(max) to all other coord
        last_touch_coord = np.reshape(touches[[1,2],-1],(2,1))
        dist = np.linalg.norm(data[[1,2],:]-last_touch_coord, axis=0)
        
        # Remove all coord within "distance_th" distance from coord(max)
        ind = np.argwhere(dist < distance_th)
        data = np.delete(data, ind, axis=1)

        # Select and store the next max 
        if data.size > 0:
            touches = np.append(touches, np.reshape(data[:,0],(3,1)), axis = 1)
    
    return touches

def test_model(X_data, Y_data):
    """
    Tests the saved model's accuracy on a given data set. Evaluates precision/recall and F1 score.
    
    Arguments:
    X_data -- input data, of shape [m, n_Hx, n_Wx, n_Cx]
    Y_data -- data labels, of shape [m, n_Hy, n_Wy, n_Cy]
    """
    
    # Load the weights
    w = h5py.File('weights_v4.h5', 'r')
    W1 = np.array(w['W1'][:])
    b1 = np.array(w['b1'][:])
    W2a = np.array(w['W2a'][:])
    b2a = np.array(w['b2a'][:])
    W2b = np.array(w['W2b'][:])
    b2b = np.array(w['b2b'][:])
    W3a = np.array(w['W3a'][:])
    b3a = np.array(w['b3a'][:])
    W3b = np.array(w['W3b'][:])
    b3b = np.array(w['b3b'][:])
    W4 = np.array(w['W4'][:])
    b4 = np.array(w['b4'][:])
    w.close()
    
    # Get data dimensions
    (m, n_H0, n_W0, n_C0) = X_data.shape
    
    # Create Placeholder of the correct shape
    X = tf.placeholder(tf.float32, shape = (None, n_H0, n_W0, n_C0))

    # Initialize parameters
    parameters = {"W1": W1, "b1": b1,
                  "W2a": W2a, "b2a": b2a,
                  "W2b": W2b, "b2b": b2b,
                  "W3a": W3a, "b3a": b3a,
                  "W3b": W3b, "b3b": b3b,
                  "W4": W4, "b4": b4}
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    _,_,_,Z4 = forward_propagation_v4(X, parameters)

    # Initialize the accuracy metrics
    error_distance_sum = 0.                           # accumulates the distance error for true positive detections
    TP = 0                                            
    FP = 0
    FN = 0
    dist_th_nms = 1.8                                 # distance threshold for Non-Max Suppression
    score_th_nms = 50                                 # score threshold for Non-Max Suppression
    dist_th_correct = 0.9                             # TP only if distance between predicted and real coordinates < dist_th_correct
    bad_frames = []                                   # keep track of data indexes where a wrong prediction was made (for error analysis)
    
    # Initialize all the variables globally
    init = tf.global_variables_initializer()
    
    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)

        # Evaluate the output of the forward propagation
        z = Z4.eval({X: X_data})
        
        np.set_printoptions(suppress=True)
        for i in range(m):
            # Store confidence matrix
            touch_presence = np.reshape(np.round(100/(1 + np.exp(-z[i,:,:,0]))), (8,16))
            # Store predicted X coordinates
            X_coord_pred = 2 * (np.reshape(z[i,:,:,1], (8,16)) + np.array(range(16))) - 0.5
            # Store predicted Y coordinates
            Y_coord_pred = 2 * (np.reshape(z[i,:,:,2], (8,16)) + np.reshape(np.array(range(8)), (8,1))) - 0.5
            # Store predicted touches
            touches_pred = non_max_suppression(touch_presence, X_coord_pred, Y_coord_pred, score_th_nms, dist_th_nms)
            
            touch_presence_real = np.reshape(np.round(100*Y_data[i,:,:,0]), (8,16))
            # Store real X coordinates
            X_coord_real = 2 * (np.reshape(Y_data[i,:,:,1], (8,16)) + np.array(range(16))) - 0.5
            # Store real Y coordinates
            Y_coord_real = 2 * (np.reshape(Y_data[i,:,:,2], (8,16)) + np.reshape(np.array(range(8)), (8,1))) - 0.5
            # Store real touches
            touches_real = non_max_suppression(touch_presence_real, X_coord_real, Y_coord_real, score_th_nms, 0.5)
            
            # Extract accuracy metrics by matching predicted and real touches
            dist_tmp, TP_tmp, FP_tmp, FN_tmp = error_distance(touches_real, touches_pred, dist_th_correct)
            
            # '0': good frame, '1': frame with FN, '2': frame with FP, '3': frame with both FP and FN
            bad_frames.append((FN_tmp>0)*1+(FP_tmp>0)*2)
            
            # Update accuracy metrics for the entire set
            error_distance_sum += dist_tmp
            TP += TP_tmp
            FP += FP_tmp
            FN += FN_tmp
            
            # If there are few input data, print predicted and labelled touches for each (for error analysis)
            if m < 10:    
                print("Identified touches :\n" + str(touches_pred))   
                print("Labelled touches :\n" + str(touches_real))
                
        # Compute accuracy metrics        
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1_score = 2 * precision * recall / (precision + recall)
        
        # If there are many input data, print the input set accuracy and indexes of wrongly predicted frames (for error analysis)
        if m >= 10:
            print("Precision = " + str(np.round(10000 * precision) /10000))
            print("Recall = " + str(np.round(10000 * recall) /10000))
            print("F1 score = " + str(np.round(10000 * F1_score) /10000))
            if len([i for i,val in enumerate(bad_frames) if val == 2])<100:
                print("frames with FN only:\n" + str([i for i,val in enumerate(bad_frames) if val == 1]))
            else:
                print(">100 frames with FN only")
            if len([i for i,val in enumerate(bad_frames) if val == 2])<100:
                print("frames with FP only:\n" + str([i for i,val in enumerate(bad_frames) if val == 2]))
            else:
                print(">100 frames with FP only")
            if len([i for i,val in enumerate(bad_frames) if val == 3])<100:
                print("frames with both FN and FP:\n" + str([i for i,val in enumerate(bad_frames) if val == 3]))
            else:
                print(">100 frames with both FN and FP")
                
        # If there's at least 1 correct prediction, compute the mean error distance        
        if TP > 0:
            print("Mean error distance = " + str(error_distance_sum/TP))    
        else:
            print("No real touches found")

def error_distance(touches_real, touches_pred, dist_th):
    """
    Uses combinatorial optimization to pair the predicted and labelled touches following the conditions:
    1) For each pair, the distance between the predicted and labelled coordinates must be <dist_th.
    2) The total error distance between the coordinates of paired touches is minimized.
    

    Arguments:
    touches_real -- array of labelled touches, of shape (3, number of real touches)
    touches_pred -- array of predicted touches, of shape (3, number of predicted touches)
    dist_th -- distance threshold for pairing a predicted touch with a labelled one 
    
    Returns:
    dist_TP -- sum of error distances for True Positives (correct prediction = paired touches)
    TP -- number of True Positives (correctly predicted touches)
    FP -- number of False Positives (wrongly predicted touches)
    FN -- number of False Negatives (missed labelled touches)
    """
    nt_real = touches_real.shape[1]                  # get number of predicted and labelled (real) touches
    nt_pred = touches_pred.shape[1]
    TP = 0
    FP = 0
    FN = 0
    
    # check if there is the same number of predicted and labelled touches (if not, increment FP or FN by the difference)
    # pad the smaller array (touches_pred or touches_real) with large numbers to make them equal before using "linear_sum_assignment"
    if nt_real > nt_pred:
        nt_delta = nt_real - nt_pred
        touches_pred = np.hstack((touches_pred, 100.*np.ones((3, nt_delta))))
        FP -= nt_delta
    if nt_real < nt_pred:
        nt_delta = nt_pred - nt_real
        touches_real = np.hstack((touches_real, 100.*np.ones((3, nt_delta))))
        FN -= nt_delta
    
    # compute the distance (cost) matrix of all possible pairings
    dist_mat = cdist(touches_real[1:3,:].T, touches_pred[1:3,:].T)
    
    # solve the linear sum assignment problem
    row_ind, col_ind = linear_sum_assignment(dist_mat)
    
    # get the array of distances (from the optimal pairs)
    dist_opt = dist_mat[row_ind, col_ind]
    
    # find the number of TP by removing pairs with distance > dist_th
    TP = dist_opt[dist_opt <= dist_th].size
    
    # compute the sum of error distances for all TP
    dist_TP = np.sum(dist_opt[dist_opt <= dist_th])
    
    # each filtered pair (distance > dist_th) is equivalent to one FP and one FN (Note: padded (dummy) touches were factored in above) 
    FP += dist_opt[dist_opt > dist_th].size
    FN += dist_opt[dist_opt > dist_th].size
    
    return dist_TP, TP, FP, FN      

def test_model_during_training(z, Y_data):
    """
    Is used during training to compute the precision, recall and mean error distance on the train/test set.
    TP, FP and FN are computed by comparing all predicted touches with the labelled touches. 

    Arguments:
    z -- output of the model
    Y_data -- data labels
    
    Returns:
    precision -- computed as TP / (TP + FP)
    recall -- computed as TP / (TP + FN)
    mean_error_distance -- in the case of TP, the mean error distance is computed between the predicted and labelled coordinates
    """
    
    # Get number of data
    m = Y_data.shape[0]
    
    # Initialize the accuracy metrics
    error_distance_sum = 0.
    TP = 0
    FP = 0
    FN = 0
    dist_th_nms = 1.8
    score_th_nms = 0
    dist_th_correct = 0.9

    for i in range(m):
        # Store confidence matrix
            touch_presence = np.reshape(np.round(100*z[i,:,:,0]), (8,16))
            # Store predicted X coordinates
            X_coord_pred = 2 * (np.reshape(z[i,:,:,1], (8,16)) + np.array(range(16))) - 0.5
            # Store predicted Y coordinates
            Y_coord_pred = 2 * (np.reshape(z[i,:,:,2], (8,16)) + np.reshape(np.array(range(8)), (8,1))) - 0.5
            # Store predicted touches
            touches_pred = non_max_suppression(touch_presence, X_coord_pred, Y_coord_pred, score_th_nms, dist_th_nms)
            
            touch_presence_real = np.reshape(np.round(100*Y_data[i,:,:,0]), (8,16))
            # Store real X coordinates
            X_coord_real = 2 * (np.reshape(Y_data[i,:,:,1], (8,16)) + np.array(range(16))) - 0.5
            # Store real Y coordinates
            Y_coord_real = 2 * (np.reshape(Y_data[i,:,:,2], (8,16)) + np.reshape(np.array(range(8)), (8,1))) - 0.5
            # Store real touches
            touches_real = non_max_suppression(touch_presence_real, X_coord_real, Y_coord_real, 50, 0.5)
            
            # Extract accuracy metrics by matching predicted and real touches
            dist_tmp, TP_tmp, FP_tmp, FN_tmp = error_distance(touches_real, touches_pred, dist_th_correct)
            
            # Update accuracy metrics for the entire set
            error_distance_sum += dist_tmp
            TP += TP_tmp
            FP += FP_tmp
            FN += FN_tmp
            
    precision = np.round(10000 * TP / (TP + FP))/10000
    recall = np.round(10000 * TP / (TP + FN))/10000
    #F1_score = 2 * precision * recall / (precision + recall)
    mean_error_distance = error_distance_sum/(TP+1e-8)
            
    return precision, recall, mean_error_distance
