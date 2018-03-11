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
    Compute focal loss for predictions. Focal loss formula:
            FL = - alpha_obj * y * (1-p)^gamma * log(p) - alpha_noobj * (1-y) * p^gamma * log(1-p)
                 ,where alpha_obj, alpha_noobj, gamma = TBD, p = sigmoid(prediction_tensor), y = target_tensor,
                 
    Arguments:
    prediction_tensor -- float tensor of shape [batch_size, num_anchors, num_classes] 
                         representing the predicted logits
    target_tensor -- float tensor of shape [batch_size, num_anchors, num_classes] 
                     representing the ground truth labels
    alpha_obj -- scalar tensor for weighing the classification cost when a touch is present
    alpha_noobj -- scalar tensor for weighing the classification cost when a touch is not present
    gamma -- scalar tensor for weighing the focall loss penalty for misclassification
     
    Returns:
    loss -- a tensor representing the value of the loss function
    """
    sigmoid_p = tf.nn.sigmoid(prediction_tensor)
    cost_touch_present = tf.multiply(target_tensor, 
                                     tf.multiply(tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)), 
                                                 tf.pow(1 - sigmoid_p, gamma)))
    cost_touch_not_present = tf.multiply(1 - target_tensor, 
                                         tf.multiply(tf.log(tf.clip_by_value(1 - sigmoid_p, 1e-8, 1.0)), 
                                                     tf.pow(sigmoid_p, gamma)))
    per_entry_cross_ent = - alpha_obj * cost_touch_present - alpha_noobj * cost_touch_not_present
    
    return tf.reduce_sum(per_entry_cross_ent)


def convert_to_one_hot(Y, C):
    """
    Converts a class label array to a one hot matrix, where class values are changed to a sparse one hot representation.
    
    Arguments:
    Y -- array containing the class labels
    C -- number of classes; the depth of the one hot dimension
    
    Returns:
    Y -- one hot matrix
    """
    
    Y = np.eye(C)[Y.reshape(-1)].T
    
    return Y


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

