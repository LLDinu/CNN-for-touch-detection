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
