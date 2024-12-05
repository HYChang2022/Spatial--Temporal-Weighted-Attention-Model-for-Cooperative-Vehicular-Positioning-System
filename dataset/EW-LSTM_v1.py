# -*- coding: utf-8 -*-
"""
Created on Sat May  6 17:13:10 2023
0506 DNN_v5 (time slot)LSTM
@author: 826BK2023
"""
import math
import tensorflow as tf
from tensorflow import keras
import numpy as np
import csv
import estimate_EW_LSTM_v1
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
import statistics 
import matplotlib.pyplot as plt
import scipy.io as sio
import logging



environment = '4.7_2.2'
number_epochs = 500
batch_size_n = 128
time_slot_N = 5
time_slot = 100000
number_test = int(time_slot*time_slot_N)
vehicle_number = 5
address = f'D:\\EW-LSTM\\GPS_estimate\\Code\\v2\\DATA\\{environment}_w_error\\velocity_100_sensor_1x'
for time_slot_steps in range(5,6):
    
    data_size = 0
    with open(f'{address}\\train_data_mt_{time_slot_N}_{environment}_v{vehicle_number}.csv', 'r') as f:
        reader = csv.reader(f)
        data = [list(map(float, row)) for row in reader]
    # 將Python list轉換為NumPy array
    arr1 = np.array(data)
    data_size = arr1.shape[0]
    
    # 五輛車估計的目標車輛位置 estimate_target_test   
    with open(f'{address}\\train_estimate_target_mt_{time_slot_N}_{environment}_v{vehicle_number}.csv', newline='') as file:
        reader = csv.reader(file)
        data = [list(map(float, row)) for row in reader]
    arr2 = np.array(data)
        #%%
    for i in range(5):
        arr2[:, 2*i] = (arr2[:, 2*i]+arr2[:, 0])/2
        arr2[:, 2*i+1] = (arr2[:, 2*i+1]+arr2[:, 1])/2
        
    train_coordinate_x = arr1[:, 0].reshape((data_size, 1))
    train_coordinate_y = arr1[:, 1].reshape((data_size, 1))
    train_coordinate = np.concatenate((train_coordinate_x, train_coordinate_y), axis=1)
    speed = arr1[:, 35].reshape(time_slot, time_slot_N, 1)
    
    train_data = np.concatenate((train_coordinate, arr2), axis=1)
    validation_data = train_data.reshape(time_slot, time_slot_N, 12)
    validation_data = np.concatenate((validation_data, speed), axis=2)
    
    train_et = arr1[:, 10:36].reshape(time_slot, time_slot_N, 26)

    #%%
    train_et = train_et[:, 0:time_slot_steps, :]
    validation_data = validation_data[:, 0:time_slot_steps, :]

    #%% # 定義model

    @tf.function
    def custom_loss(y_true, y_pred):
       
        y_true=tf.reshape(y_true, (batch_size_n, time_slot_steps, 13))
        loss = tf.zeros([])
        gps_pred = tf.TensorArray(tf.float32, size=time_slot_steps)
        for i in range(time_slot_steps):
            if i == 0:
                gps_pred_x_t0 = tf.expand_dims(y_true[:, 0, 2]*y_pred[:, 0, 0] 
                                               + y_true[:, 0, 4]*y_pred[:, 0, 2] 
                                               + y_true[:, 0, 6]*y_pred[:, 0, 4] 
                                               + y_true[:, 0, 8]*y_pred[:, 0, 6] 
                                               + y_true[:, 0, 10]*y_pred[:, 0, 8], axis=1)
                gps_pred_y_t0 = tf.expand_dims(y_true[:, 0, 3]*y_pred[:, 0, 1] 
                                               + y_true[:, 0, 5]*y_pred[:, 0, 3]
                                               + y_true[:, 0, 7]*y_pred[:, 0, 5] 
                                               +y_true[:, 0, 9]*y_pred[:, 0, 7] 
                                               + y_true[:, 0, 11]*y_pred[:, 0, 9], axis=1)
                gps_pred = gps_pred.write(0, tf.concat([gps_pred_x_t0, gps_pred_y_t0], axis=1))  
            #"""    
            elif i == 1:
                gps_pred_x = tf.expand_dims((gps_pred_x_t0[:, 0]+y_true[:, i, 12]*y_pred[:, i, 12])*y_pred[:, i, 10]
                                            + y_true[:, i, 2]*y_pred[:, i, 0] 
                                            + y_true[:, i, 4]*y_pred[:, i, 2] 
                                            + y_true[:, i, 6]*y_pred[:, i, 4] 
                                            + y_true[:, i, 8]*y_pred[:, i, 6] 
                                            + y_true[:, i, 10]*y_pred[:, i, 8], axis=1)  
                
                gps_pred_y = tf.expand_dims((gps_pred_y_t0[:, 0]+y_true[:, i, 12]*y_pred[:, i, 13])*y_pred[:, i, 11]
                                            + y_true[:, i, 3]*y_pred[:, i, 1] 
                                            + y_true[:, i, 5]*y_pred[:, i, 3] 
                                            + y_true[:, i, 7]*y_pred[:, i, 5] 
                                            + y_true[:, i, 9]*y_pred[:, i, 7] 
                                            + y_true[:, i, 11]*y_pred[:, i, 9], axis=1)
                gps_pred = gps_pred.write(i, tf.concat([gps_pred_x, gps_pred_y], axis=1))
            else:
                gps_pred_x = tf.expand_dims((gps_pred.read(i-1)[:, 0]+y_true[:, i, 12]*y_pred[:, i, 12] )*y_pred[:, i, 10] 
                                            + y_true[:, i, 2]*y_pred[:, i, 0] 
                                            + y_true[:, i, 4]*y_pred[:, i, 2] 
                                            + y_true[:, i, 6]*y_pred[:, i, 4] 
                                            + y_true[:, i, 8]*y_pred[:, i, 6] 
                                            + y_true[:, i, 10]*y_pred[:, i, 8], axis=1)
                gps_pred_y = tf.expand_dims((gps_pred.read(i-1)[:, 1]+y_true[:, i, 12]*y_pred[:, i, 13])* y_pred[:, i, 11] 
                                            + y_true[:, i, 3]*y_pred[:, i, 1] 
                                            + y_true[:, i, 5]*y_pred[:, i, 3] 
                                            + y_true[:, i, 7]*y_pred[:, i, 5] 
                                            + y_true[:, i, 9]*y_pred[:, i, 7] 
                                            + y_true[:, i, 11]*y_pred[:, i, 9], axis=1)
                gps_pred = gps_pred.write(i, tf.concat([gps_pred_x, gps_pred_y], axis=1))
            #""" 
            loss += K.sqrt(K.square(gps_pred.read(i)[:, 0] - y_true[:, i, 0]) 
                          + K.square(gps_pred.read(i)[:, 1] - y_true[:, i, 1]))
            if i==0:
                loss_0 = K.sqrt(K.square(gps_pred.read(i)[:, 0] - y_true[:, i, 0]) 
                          + K.square(gps_pred.read(i)[:, 1] - y_true[:, i, 1]))
            elif i==1:
                loss_1 = K.sqrt(K.square(gps_pred.read(i)[:, 0] - y_true[:, i, 0]) 
                          + K.square(gps_pred.read(i)[:, 1] - y_true[:, i, 1]))
            elif i==2:
                loss_2 = K.sqrt(K.square(gps_pred.read(i)[:, 0] - y_true[:, i, 0]) 
                          + K.square(gps_pred.read(i)[:, 1] - y_true[:, i, 1]))
            elif i==3:
                loss_3 = K.sqrt(K.square(gps_pred.read(i)[:, 0] - y_true[:, i, 0]) 
                          + K.square(gps_pred.read(i)[:, 1] - y_true[:, i, 1]))
            elif i==4:
                loss_4 = K.sqrt(K.square(gps_pred.read(i)[:, 0] - y_true[:, i, 0]) 
                          + K.square(gps_pred.read(i)[:, 1] - y_true[:, i, 1]))
          # Normalize the loss by the number of time steps
    
        return loss, loss_0, loss_1, loss_2, loss_3, loss_4
    
    @tf.function
    def loss_wrapper(y_true, y_pred):
        loss, loss_0, loss_1, loss_2, loss_3, loss_4 = custom_loss(y_true, y_pred)
        # logging.info(f"Loss 0: {loss_0}, Loss 1: {loss_1}, Loss 2: {loss_2}, Loss 3: {loss_3}, Loss 4: {loss_4}")
        return loss
    
    @tf.function
    def custom_loss0(y_true, y_pred):
       
        y_true=tf.reshape(y_true, (batch_size_n, time_slot_steps, 13))
        loss = tf.zeros([])
        gps_pred = tf.TensorArray(tf.float32, size=time_slot_steps)
        for i in range(time_slot_steps):
            if i == 0:
                gps_pred_x_t0 = tf.expand_dims(y_true[:, 0, 2]*y_pred[:, 0, 0] 
                                               + y_true[:, 0, 4]*y_pred[:, 0, 2] 
                                               + y_true[:, 0, 6]*y_pred[:, 0, 4] 
                                               + y_true[:, 0, 8]*y_pred[:, 0, 6] 
                                               + y_true[:, 0, 10]*y_pred[:, 0, 8], axis=1)
                gps_pred_y_t0 = tf.expand_dims(y_true[:, 0, 3]*y_pred[:, 0, 1] 
                                               + y_true[:, 0, 5]*y_pred[:, 0, 3]
                                               + y_true[:, 0, 7]*y_pred[:, 0, 5] 
                                               +y_true[:, 0, 9]*y_pred[:, 0, 7] 
                                               + y_true[:, 0, 11]*y_pred[:, 0, 9], axis=1)
                gps_pred = gps_pred.write(0, tf.concat([gps_pred_x_t0, gps_pred_y_t0], axis=1))  
            #"""    
            elif i == 1:
                gps_pred_x = tf.expand_dims((gps_pred_x_t0[:, 0]+y_true[:, i, 12]*y_pred[:, i, 12])*y_pred[:, i, 10]
                                            + y_true[:, i, 2]*y_pred[:, i, 0] 
                                            + y_true[:, i, 4]*y_pred[:, i, 2] 
                                            + y_true[:, i, 6]*y_pred[:, i, 4] 
                                            + y_true[:, i, 8]*y_pred[:, i, 6] 
                                            + y_true[:, i, 10]*y_pred[:, i, 8], axis=1)  
                
                gps_pred_y = tf.expand_dims((gps_pred_y_t0[:, 0]+y_true[:, i, 12]*y_pred[:, i, 13])*y_pred[:, i, 11]
                                            + y_true[:, i, 3]*y_pred[:, i, 1] 
                                            + y_true[:, i, 5]*y_pred[:, i, 3] 
                                            + y_true[:, i, 7]*y_pred[:, i, 5] 
                                            + y_true[:, i, 9]*y_pred[:, i, 7] 
                                            + y_true[:, i, 11]*y_pred[:, i, 9], axis=1)
                gps_pred = gps_pred.write(i, tf.concat([gps_pred_x, gps_pred_y], axis=1))
            else:
                gps_pred_x = tf.expand_dims((gps_pred.read(i-1)[:, 0]+y_true[:, i, 12]*y_pred[:, i, 12] )*y_pred[:, i, 10] 
                                            + y_true[:, i, 2]*y_pred[:, i, 0] 
                                            + y_true[:, i, 4]*y_pred[:, i, 2] 
                                            + y_true[:, i, 6]*y_pred[:, i, 4] 
                                            + y_true[:, i, 8]*y_pred[:, i, 6] 
                                            + y_true[:, i, 10]*y_pred[:, i, 8], axis=1)
                gps_pred_y = tf.expand_dims((gps_pred.read(i-1)[:, 1]+y_true[:, i, 12]*y_pred[:, i, 13])* y_pred[:, i, 11] 
                                            + y_true[:, i, 3]*y_pred[:, i, 1] 
                                            + y_true[:, i, 5]*y_pred[:, i, 3] 
                                            + y_true[:, i, 7]*y_pred[:, i, 5] 
                                            + y_true[:, i, 9]*y_pred[:, i, 7] 
                                            + y_true[:, i, 11]*y_pred[:, i, 9], axis=1)
                gps_pred = gps_pred.write(i, tf.concat([gps_pred_x, gps_pred_y], axis=1))
            #""" 
            loss = K.sqrt(K.square(gps_pred.read(i)[:, 0] - y_true[:, i, 0]) 
                          + K.square(gps_pred.read(i)[:, 1] - y_true[:, i, 1]))
            if i==0:
                loss_0 = K.sqrt(K.square(gps_pred.read(i)[:, 0] - y_true[:, i, 0]) 
                          + K.square(gps_pred.read(i)[:, 1] - y_true[:, i, 1]))
            elif i==1:
                loss_1 = K.sqrt(K.square(gps_pred.read(i)[:, 0] - y_true[:, i, 0]) 
                          + K.square(gps_pred.read(i)[:, 1] - y_true[:, i, 1]))
            elif i==2:
                loss_2 = K.sqrt(K.square(gps_pred.read(i)[:, 0] - y_true[:, i, 0]) 
                          + K.square(gps_pred.read(i)[:, 1] - y_true[:, i, 1]))
            elif i==3:
                loss_3 = K.sqrt(K.square(gps_pred.read(i)[:, 0] - y_true[:, i, 0]) 
                          + K.square(gps_pred.read(i)[:, 1] - y_true[:, i, 1]))
            elif i==4:
                loss_4 = K.sqrt(K.square(gps_pred.read(i)[:, 0] - y_true[:, i, 0]) 
                          + K.square(gps_pred.read(i)[:, 1] - y_true[:, i, 1]))
        return loss_0
    @tf.function
    def custom_loss1(y_true, y_pred):
       
        y_true=tf.reshape(y_true, (batch_size_n, time_slot_steps, 13))
        loss = tf.zeros([])
        gps_pred = tf.TensorArray(tf.float32, size=time_slot_steps)
        for i in range(time_slot_steps):
            if i == 0:
                gps_pred_x_t0 = tf.expand_dims(y_true[:, 0, 2]*y_pred[:, 0, 0] 
                                               + y_true[:, 0, 4]*y_pred[:, 0, 2] 
                                               + y_true[:, 0, 6]*y_pred[:, 0, 4] 
                                               + y_true[:, 0, 8]*y_pred[:, 0, 6] 
                                               + y_true[:, 0, 10]*y_pred[:, 0, 8], axis=1)
                gps_pred_y_t0 = tf.expand_dims(y_true[:, 0, 3]*y_pred[:, 0, 1] 
                                               + y_true[:, 0, 5]*y_pred[:, 0, 3]
                                               + y_true[:, 0, 7]*y_pred[:, 0, 5] 
                                               +y_true[:, 0, 9]*y_pred[:, 0, 7] 
                                               + y_true[:, 0, 11]*y_pred[:, 0, 9], axis=1)
                gps_pred = gps_pred.write(0, tf.concat([gps_pred_x_t0, gps_pred_y_t0], axis=1))  
            #"""    
            elif i == 1:
                gps_pred_x = tf.expand_dims((gps_pred_x_t0[:, 0]+y_true[:, i, 12]*y_pred[:, i, 12])*y_pred[:, i, 10]
                                            + y_true[:, i, 2]*y_pred[:, i, 0] 
                                            + y_true[:, i, 4]*y_pred[:, i, 2] 
                                            + y_true[:, i, 6]*y_pred[:, i, 4] 
                                            + y_true[:, i, 8]*y_pred[:, i, 6] 
                                            + y_true[:, i, 10]*y_pred[:, i, 8], axis=1)  
                
                gps_pred_y = tf.expand_dims((gps_pred_y_t0[:, 0]+y_true[:, i, 12]*y_pred[:, i, 13])*y_pred[:, i, 11]
                                            + y_true[:, i, 3]*y_pred[:, i, 1] 
                                            + y_true[:, i, 5]*y_pred[:, i, 3] 
                                            + y_true[:, i, 7]*y_pred[:, i, 5] 
                                            + y_true[:, i, 9]*y_pred[:, i, 7] 
                                            + y_true[:, i, 11]*y_pred[:, i, 9], axis=1)
                gps_pred = gps_pred.write(i, tf.concat([gps_pred_x, gps_pred_y], axis=1))
            else:
                gps_pred_x = tf.expand_dims((gps_pred.read(i-1)[:, 0]+y_true[:, i, 12]*y_pred[:, i, 12] )*y_pred[:, i, 10] 
                                            + y_true[:, i, 2]*y_pred[:, i, 0] 
                                            + y_true[:, i, 4]*y_pred[:, i, 2] 
                                            + y_true[:, i, 6]*y_pred[:, i, 4] 
                                            + y_true[:, i, 8]*y_pred[:, i, 6] 
                                            + y_true[:, i, 10]*y_pred[:, i, 8], axis=1)
                gps_pred_y = tf.expand_dims((gps_pred.read(i-1)[:, 1]+y_true[:, i, 12]*y_pred[:, i, 13])* y_pred[:, i, 11] 
                                            + y_true[:, i, 3]*y_pred[:, i, 1] 
                                            + y_true[:, i, 5]*y_pred[:, i, 3] 
                                            + y_true[:, i, 7]*y_pred[:, i, 5] 
                                            + y_true[:, i, 9]*y_pred[:, i, 7] 
                                            + y_true[:, i, 11]*y_pred[:, i, 9], axis=1)
                gps_pred = gps_pred.write(i, tf.concat([gps_pred_x, gps_pred_y], axis=1))
            #""" 
            loss = K.sqrt(K.square(gps_pred.read(i)[:, 0] - y_true[:, i, 0]) 
                          + K.square(gps_pred.read(i)[:, 1] - y_true[:, i, 1]))
            if i==0:
                loss_0 = K.sqrt(K.square(gps_pred.read(i)[:, 0] - y_true[:, i, 0]) 
                          + K.square(gps_pred.read(i)[:, 1] - y_true[:, i, 1]))
            elif i==1:
                loss_1 = K.sqrt(K.square(gps_pred.read(i)[:, 0] - y_true[:, i, 0]) 
                          + K.square(gps_pred.read(i)[:, 1] - y_true[:, i, 1]))
            elif i==2:
                loss_2 = K.sqrt(K.square(gps_pred.read(i)[:, 0] - y_true[:, i, 0]) 
                          + K.square(gps_pred.read(i)[:, 1] - y_true[:, i, 1]))
            elif i==3:
                loss_3 = K.sqrt(K.square(gps_pred.read(i)[:, 0] - y_true[:, i, 0]) 
                          + K.square(gps_pred.read(i)[:, 1] - y_true[:, i, 1]))
            elif i==4:
                loss_4 = K.sqrt(K.square(gps_pred.read(i)[:, 0] - y_true[:, i, 0]) 
                          + K.square(gps_pred.read(i)[:, 1] - y_true[:, i, 1]))    
        return loss_1
    @tf.function
    def custom_loss2(y_true, y_pred):
       
        y_true=tf.reshape(y_true, (batch_size_n, time_slot_steps, 13))
        loss = tf.zeros([])
        gps_pred = tf.TensorArray(tf.float32, size=time_slot_steps)
        for i in range(time_slot_steps):
            if i == 0:
                gps_pred_x_t0 = tf.expand_dims(y_true[:, 0, 2]*y_pred[:, 0, 0] 
                                               + y_true[:, 0, 4]*y_pred[:, 0, 2] 
                                               + y_true[:, 0, 6]*y_pred[:, 0, 4] 
                                               + y_true[:, 0, 8]*y_pred[:, 0, 6] 
                                               + y_true[:, 0, 10]*y_pred[:, 0, 8], axis=1)
                gps_pred_y_t0 = tf.expand_dims(y_true[:, 0, 3]*y_pred[:, 0, 1] 
                                               + y_true[:, 0, 5]*y_pred[:, 0, 3]
                                               + y_true[:, 0, 7]*y_pred[:, 0, 5] 
                                               +y_true[:, 0, 9]*y_pred[:, 0, 7] 
                                               + y_true[:, 0, 11]*y_pred[:, 0, 9], axis=1)
                gps_pred = gps_pred.write(0, tf.concat([gps_pred_x_t0, gps_pred_y_t0], axis=1))  
            #"""    
            elif i == 1:
                gps_pred_x = tf.expand_dims((gps_pred_x_t0[:, 0]+y_true[:, i, 12]*y_pred[:, i, 12])*y_pred[:, i, 10]
                                            + y_true[:, i, 2]*y_pred[:, i, 0] 
                                            + y_true[:, i, 4]*y_pred[:, i, 2] 
                                            + y_true[:, i, 6]*y_pred[:, i, 4] 
                                            + y_true[:, i, 8]*y_pred[:, i, 6] 
                                            + y_true[:, i, 10]*y_pred[:, i, 8], axis=1)  
                
                gps_pred_y = tf.expand_dims((gps_pred_y_t0[:, 0]+y_true[:, i, 12]*y_pred[:, i, 13])*y_pred[:, i, 11]
                                            + y_true[:, i, 3]*y_pred[:, i, 1] 
                                            + y_true[:, i, 5]*y_pred[:, i, 3] 
                                            + y_true[:, i, 7]*y_pred[:, i, 5] 
                                            + y_true[:, i, 9]*y_pred[:, i, 7] 
                                            + y_true[:, i, 11]*y_pred[:, i, 9], axis=1)
                gps_pred = gps_pred.write(i, tf.concat([gps_pred_x, gps_pred_y], axis=1))
            else:
                gps_pred_x = tf.expand_dims((gps_pred.read(i-1)[:, 0]+y_true[:, i, 12]*y_pred[:, i, 12] )*y_pred[:, i, 10] 
                                            + y_true[:, i, 2]*y_pred[:, i, 0] 
                                            + y_true[:, i, 4]*y_pred[:, i, 2] 
                                            + y_true[:, i, 6]*y_pred[:, i, 4] 
                                            + y_true[:, i, 8]*y_pred[:, i, 6] 
                                            + y_true[:, i, 10]*y_pred[:, i, 8], axis=1)
                gps_pred_y = tf.expand_dims((gps_pred.read(i-1)[:, 1]+y_true[:, i, 12]*y_pred[:, i, 13])* y_pred[:, i, 11] 
                                            + y_true[:, i, 3]*y_pred[:, i, 1] 
                                            + y_true[:, i, 5]*y_pred[:, i, 3] 
                                            + y_true[:, i, 7]*y_pred[:, i, 5] 
                                            + y_true[:, i, 9]*y_pred[:, i, 7] 
                                            + y_true[:, i, 11]*y_pred[:, i, 9], axis=1)
                gps_pred = gps_pred.write(i, tf.concat([gps_pred_x, gps_pred_y], axis=1))
            #""" 
            loss = K.sqrt(K.square(gps_pred.read(i)[:, 0] - y_true[:, i, 0]) 
                          + K.square(gps_pred.read(i)[:, 1] - y_true[:, i, 1]))
            if i==0:
                loss_0 = K.sqrt(K.square(gps_pred.read(i)[:, 0] - y_true[:, i, 0]) 
                          + K.square(gps_pred.read(i)[:, 1] - y_true[:, i, 1]))
            elif i==1:
                loss_1 = K.sqrt(K.square(gps_pred.read(i)[:, 0] - y_true[:, i, 0]) 
                          + K.square(gps_pred.read(i)[:, 1] - y_true[:, i, 1]))
            elif i==2:
                loss_2 = K.sqrt(K.square(gps_pred.read(i)[:, 0] - y_true[:, i, 0]) 
                          + K.square(gps_pred.read(i)[:, 1] - y_true[:, i, 1]))
            elif i==3:
                loss_3 = K.sqrt(K.square(gps_pred.read(i)[:, 0] - y_true[:, i, 0]) 
                          + K.square(gps_pred.read(i)[:, 1] - y_true[:, i, 1]))
            elif i==4:
                loss_4 = K.sqrt(K.square(gps_pred.read(i)[:, 0] - y_true[:, i, 0]) 
                          + K.square(gps_pred.read(i)[:, 1] - y_true[:, i, 1]))    
        return loss_2
    @tf.function
    def custom_loss3(y_true, y_pred):
       
        y_true=tf.reshape(y_true, (batch_size_n, time_slot_steps, 13))
        loss = tf.zeros([])
        gps_pred = tf.TensorArray(tf.float32, size=time_slot_steps)
        for i in range(time_slot_steps):
            if i == 0:
                gps_pred_x_t0 = tf.expand_dims(y_true[:, 0, 2]*y_pred[:, 0, 0] 
                                               + y_true[:, 0, 4]*y_pred[:, 0, 2] 
                                               + y_true[:, 0, 6]*y_pred[:, 0, 4] 
                                               + y_true[:, 0, 8]*y_pred[:, 0, 6] 
                                               + y_true[:, 0, 10]*y_pred[:, 0, 8], axis=1)
                gps_pred_y_t0 = tf.expand_dims(y_true[:, 0, 3]*y_pred[:, 0, 1] 
                                               + y_true[:, 0, 5]*y_pred[:, 0, 3]
                                               + y_true[:, 0, 7]*y_pred[:, 0, 5] 
                                               +y_true[:, 0, 9]*y_pred[:, 0, 7] 
                                               + y_true[:, 0, 11]*y_pred[:, 0, 9], axis=1)
                gps_pred = gps_pred.write(0, tf.concat([gps_pred_x_t0, gps_pred_y_t0], axis=1))  
            #"""    
            elif i == 1:
                gps_pred_x = tf.expand_dims((gps_pred_x_t0[:, 0]+y_true[:, i, 12]*y_pred[:, i, 12])*y_pred[:, i, 10]
                                            + y_true[:, i, 2]*y_pred[:, i, 0] 
                                            + y_true[:, i, 4]*y_pred[:, i, 2] 
                                            + y_true[:, i, 6]*y_pred[:, i, 4] 
                                            + y_true[:, i, 8]*y_pred[:, i, 6] 
                                            + y_true[:, i, 10]*y_pred[:, i, 8], axis=1)  
                
                gps_pred_y = tf.expand_dims((gps_pred_y_t0[:, 0]+y_true[:, i, 12]*y_pred[:, i, 13])*y_pred[:, i, 11]
                                            + y_true[:, i, 3]*y_pred[:, i, 1] 
                                            + y_true[:, i, 5]*y_pred[:, i, 3] 
                                            + y_true[:, i, 7]*y_pred[:, i, 5] 
                                            + y_true[:, i, 9]*y_pred[:, i, 7] 
                                            + y_true[:, i, 11]*y_pred[:, i, 9], axis=1)
                gps_pred = gps_pred.write(i, tf.concat([gps_pred_x, gps_pred_y], axis=1))
            else:
                gps_pred_x = tf.expand_dims((gps_pred.read(i-1)[:, 0]+y_true[:, i, 12]*y_pred[:, i, 12] )*y_pred[:, i, 10] 
                                            + y_true[:, i, 2]*y_pred[:, i, 0] 
                                            + y_true[:, i, 4]*y_pred[:, i, 2] 
                                            + y_true[:, i, 6]*y_pred[:, i, 4] 
                                            + y_true[:, i, 8]*y_pred[:, i, 6] 
                                            + y_true[:, i, 10]*y_pred[:, i, 8], axis=1)
                gps_pred_y = tf.expand_dims((gps_pred.read(i-1)[:, 1]+y_true[:, i, 12]*y_pred[:, i, 13])* y_pred[:, i, 11] 
                                            + y_true[:, i, 3]*y_pred[:, i, 1] 
                                            + y_true[:, i, 5]*y_pred[:, i, 3] 
                                            + y_true[:, i, 7]*y_pred[:, i, 5] 
                                            + y_true[:, i, 9]*y_pred[:, i, 7] 
                                            + y_true[:, i, 11]*y_pred[:, i, 9], axis=1)
                gps_pred = gps_pred.write(i, tf.concat([gps_pred_x, gps_pred_y], axis=1))
            #""" 
            loss = K.sqrt(K.square(gps_pred.read(i)[:, 0] - y_true[:, i, 0]) 
                          + K.square(gps_pred.read(i)[:, 1] - y_true[:, i, 1]))
            if i==0:
                loss_0 = K.sqrt(K.square(gps_pred.read(i)[:, 0] - y_true[:, i, 0]) 
                          + K.square(gps_pred.read(i)[:, 1] - y_true[:, i, 1]))
            elif i==1:
                loss_1 = K.sqrt(K.square(gps_pred.read(i)[:, 0] - y_true[:, i, 0]) 
                          + K.square(gps_pred.read(i)[:, 1] - y_true[:, i, 1]))
            elif i==2:
                loss_2 = K.sqrt(K.square(gps_pred.read(i)[:, 0] - y_true[:, i, 0]) 
                          + K.square(gps_pred.read(i)[:, 1] - y_true[:, i, 1]))
            elif i==3:
                loss_3 = K.sqrt(K.square(gps_pred.read(i)[:, 0] - y_true[:, i, 0]) 
                          + K.square(gps_pred.read(i)[:, 1] - y_true[:, i, 1]))
            elif i==4:
                loss_4 = K.sqrt(K.square(gps_pred.read(i)[:, 0] - y_true[:, i, 0]) 
                          + K.square(gps_pred.read(i)[:, 1] - y_true[:, i, 1]))    
        return loss_3
    @tf.function
    def custom_loss4(y_true, y_pred):
       
        y_true=tf.reshape(y_true, (batch_size_n, time_slot_steps, 13))
        loss = tf.zeros([])
        gps_pred = tf.TensorArray(tf.float32, size=time_slot_steps)
        for i in range(time_slot_steps):
            if i == 0:
                gps_pred_x_t0 = tf.expand_dims(y_true[:, 0, 2]*y_pred[:, 0, 0] 
                                               + y_true[:, 0, 4]*y_pred[:, 0, 2] 
                                               + y_true[:, 0, 6]*y_pred[:, 0, 4] 
                                               + y_true[:, 0, 8]*y_pred[:, 0, 6] 
                                               + y_true[:, 0, 10]*y_pred[:, 0, 8], axis=1)
                gps_pred_y_t0 = tf.expand_dims(y_true[:, 0, 3]*y_pred[:, 0, 1] 
                                               + y_true[:, 0, 5]*y_pred[:, 0, 3]
                                               + y_true[:, 0, 7]*y_pred[:, 0, 5] 
                                               +y_true[:, 0, 9]*y_pred[:, 0, 7] 
                                               + y_true[:, 0, 11]*y_pred[:, 0, 9], axis=1)
                gps_pred = gps_pred.write(0, tf.concat([gps_pred_x_t0, gps_pred_y_t0], axis=1))  
            #"""    
            elif i == 1:
                gps_pred_x = tf.expand_dims((gps_pred_x_t0[:, 0]+y_true[:, i, 12]*y_pred[:, i, 12])*y_pred[:, i, 10]
                                            + y_true[:, i, 2]*y_pred[:, i, 0] 
                                            + y_true[:, i, 4]*y_pred[:, i, 2] 
                                            + y_true[:, i, 6]*y_pred[:, i, 4] 
                                            + y_true[:, i, 8]*y_pred[:, i, 6] 
                                            + y_true[:, i, 10]*y_pred[:, i, 8], axis=1)  
                
                gps_pred_y = tf.expand_dims((gps_pred_y_t0[:, 0]+y_true[:, i, 12]*y_pred[:, i, 13])*y_pred[:, i, 11]
                                            + y_true[:, i, 3]*y_pred[:, i, 1] 
                                            + y_true[:, i, 5]*y_pred[:, i, 3] 
                                            + y_true[:, i, 7]*y_pred[:, i, 5] 
                                            + y_true[:, i, 9]*y_pred[:, i, 7] 
                                            + y_true[:, i, 11]*y_pred[:, i, 9], axis=1)
                gps_pred = gps_pred.write(i, tf.concat([gps_pred_x, gps_pred_y], axis=1))
            else:
                gps_pred_x = tf.expand_dims((gps_pred.read(i-1)[:, 0]+y_true[:, i, 12]*y_pred[:, i, 12] )*y_pred[:, i, 10] 
                                            + y_true[:, i, 2]*y_pred[:, i, 0] 
                                            + y_true[:, i, 4]*y_pred[:, i, 2] 
                                            + y_true[:, i, 6]*y_pred[:, i, 4] 
                                            + y_true[:, i, 8]*y_pred[:, i, 6] 
                                            + y_true[:, i, 10]*y_pred[:, i, 8], axis=1)
                gps_pred_y = tf.expand_dims((gps_pred.read(i-1)[:, 1]+y_true[:, i, 12]*y_pred[:, i, 13])* y_pred[:, i, 11] 
                                            + y_true[:, i, 3]*y_pred[:, i, 1] 
                                            + y_true[:, i, 5]*y_pred[:, i, 3] 
                                            + y_true[:, i, 7]*y_pred[:, i, 5] 
                                            + y_true[:, i, 9]*y_pred[:, i, 7] 
                                            + y_true[:, i, 11]*y_pred[:, i, 9], axis=1)
                gps_pred = gps_pred.write(i, tf.concat([gps_pred_x, gps_pred_y], axis=1))
            #""" 
            loss = K.sqrt(K.square(gps_pred.read(i)[:, 0] - y_true[:, i, 0]) 
                          + K.square(gps_pred.read(i)[:, 1] - y_true[:, i, 1]))
            if i==0:
                loss_0 = K.sqrt(K.square(gps_pred.read(i)[:, 0] - y_true[:, i, 0]) 
                          + K.square(gps_pred.read(i)[:, 1] - y_true[:, i, 1]))
            elif i==1:
                loss_1 = K.sqrt(K.square(gps_pred.read(i)[:, 0] - y_true[:, i, 0]) 
                          + K.square(gps_pred.read(i)[:, 1] - y_true[:, i, 1]))
            elif i==2:
                loss_2 = K.sqrt(K.square(gps_pred.read(i)[:, 0] - y_true[:, i, 0]) 
                          + K.square(gps_pred.read(i)[:, 1] - y_true[:, i, 1]))
            elif i==3:
                loss_3 = K.sqrt(K.square(gps_pred.read(i)[:, 0] - y_true[:, i, 0]) 
                          + K.square(gps_pred.read(i)[:, 1] - y_true[:, i, 1]))
            elif i==4:
                loss_4 = K.sqrt(K.square(gps_pred.read(i)[:, 0] - y_true[:, i, 0]) 
                          + K.square(gps_pred.read(i)[:, 1] - y_true[:, i, 1]))    
        return loss_4
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(100, input_shape=(time_slot_steps, 26), return_sequences=True),#100
        #tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.LSTM(64, return_sequences=True),
        #tf.keras.layers.Dropout(0.2), 
        #tf.keras.layers.Dropout(0.2),
        #tf.keras.layers.Dense(1280, activation='relu'),
        tf.keras.layers.Dense(14, activation='linear')
    ])
    tf.keras.utils.get_custom_objects()['custom_loss'] = custom_loss
    
    adam = tf.keras.optimizers.Adam(lr=0.0001)#0.0001

    model.compile(optimizer=adam, loss=loss_wrapper, metrics=[custom_loss0, custom_loss1, custom_loss2, custom_loss3, custom_loss4])
    #model.compile(optimizer='adam', loss='custom_loss')
    
    checkpoint = ModelCheckpoint('model_weights.h5', save_weights_only=False)
    
    gps = model.fit(train_et, validation_data, 
                    epochs=number_epochs, batch_size=batch_size_n, 
                    validation_split=0.2, callbacks=[checkpoint], verbose = 1)
    
    train_loss = gps.history['loss']
    val_loss = gps.history['val_loss']
    train_loss_0 = gps.history['custom_loss0']
    val_loss_0 = gps.history['val_custom_loss0']
    train_loss_1 = gps.history['custom_loss1']
    val_loss_1 = gps.history['val_custom_loss1']
    train_loss_2 = gps.history['custom_loss2']
    val_loss_2 = gps.history['val_custom_loss2']
    train_loss_3 = gps.history['custom_loss3']
    val_loss_3 = gps.history['val_custom_loss3']
    train_loss_4 = gps.history['custom_loss4']
    val_loss_4 = gps.history['val_custom_loss4']
    #%%
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_loss, 'ro', label='Training Loss')
    plt.plot(epochs, val_loss, 'b*', label='Validation Loss')
    plt.title(f'Time Slot {time_slot_steps} Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xlim(number_epochs*0, number_epochs)
    plt.ylim(0, 100)
    plt.legend()
    plt.show()
    address2 = f"D:\\EW-LSTM\\weight\\velocity_100_sensor_1x"
    sio.savemat(f'{address2}\\EWLSTM_v2_t{time_slot_steps}_{environment}_train_loss.mat', {"train_loss": train_loss})
    sio.savemat(f'{address2}\\EWLSTM_v2_t{time_slot_steps}_{environment}_val_loss.mat', {"val_loss": val_loss})
    sio.savemat(f'{address2}\\EWLSTM_v2_t{time_slot_steps}_{environment}_train_loss_error1.mat', {"train_loss": train_loss_0})
    sio.savemat(f'{address2}\\EWLSTM_v2_t{time_slot_steps}_{environment}_val_loss_error1.mat', {"val_loss": val_loss_0})
    sio.savemat(f'{address2}\\EWLSTM_v2_t{time_slot_steps}_{environment}_train_loss_error2.mat', {"train_loss": train_loss_1})
    sio.savemat(f'{address2}\\EWLSTM_v2_t{time_slot_steps}_{environment}_val_loss_error2.mat', {"val_loss": val_loss_1})
    sio.savemat(f'{address2}\\EWLSTM_v2_t{time_slot_steps}_{environment}_train_loss_error3.mat', {"train_loss": train_loss_2})
    sio.savemat(f'{address2}\\EWLSTM_v2_t{time_slot_steps}_{environment}_val_loss_error3.mat', {"val_loss": val_loss_2})
    sio.savemat(f'{address2}\\EWLSTM_v2_t{time_slot_steps}_{environment}_train_loss_error4.mat', {"train_loss": train_loss_3})
    sio.savemat(f'{address2}\\EWLSTM_v2_t{time_slot_steps}_{environment}_val_loss_error4.mat', {"val_loss": val_loss_3})
    sio.savemat(f'{address2}\\EWLSTM_v2_t{time_slot_steps}_{environment}_train_loss_error5.mat', {"train_loss": train_loss_4})
    sio.savemat(f'{address2}\\EWLSTM_v2_t{time_slot_steps}_{environment}_val_loss_error5.mat', {"val_loss": val_loss_4})
    #%% 
    import estimate_LSTM_Weight_2
    address = f'D:\\EW-LSTM\\GPS_estimate\\Code\\v2\\DATA\\{environment}_w_error\\velocity_400'
    address2 = f"D:\\EW-LSTM\\weight\\velocity_400"
    result_x = []
    result_y = []
    x = np.zeros((time_slot, time_slot_N, 1))
    y = np.zeros((time_slot, time_slot_N, 1))  
    
    with open(f'{address}\\test_data_mt_{time_slot_N}_{environment}_v{vehicle_number}.csv', 'r') as f:
        reader = csv.reader(f)
        err1 = [list(map(float, row)) for row in reader]
    err1 = np.array(err1)
    
    with open(f'{address}\\test_estimate_target_mt_{time_slot_N}_{environment}_v{vehicle_number}.csv') as f:
        reader = csv.reader(f)
        err2 = [list(map(float, row)) for row in reader]
    err2 = np.array(err2)
    
    for i in range(5):
        err2[:, 2*i] = (err2[:, 2*i]+err2[:, 0])/2
        err2[:, 2*i+1] = (err2[:, 2*i+1]+err2[:, 1])/2
        
    speed = err1[:, 35].reshape(time_slot, time_slot_N, 1)
    test_c_x = err1[(time_slot_steps-1)::time_slot_N, 0]
    test_c_y = err1[(time_slot_steps-1)::time_slot_N, 1]
    

    test_data = err1[:, 10:36].reshape(time_slot, time_slot_N, 26)
    test_data_et = err2.reshape(time_slot, time_slot_N, 10)
    #test_data_et = np.concatenate((err2.reshape(time_slot, time_slot_steps, 10), test_data_et), axis=2)

    test_data_et = test_data_et[:, 0:time_slot_steps, :]
    test_data = test_data[:, 0:time_slot_steps, :]
    x = x[:, 0:time_slot_steps, :]
    y = y[:, 0:time_slot_steps, :]

    #%%                 
    w = model.predict(test_data)
    for i in range(time_slot_steps):
        if i == 0:
            x[:, i, 0] = (test_data_et[:, i, 0]*w[:, i, 0] 
                          + test_data_et[:, i, 2]*w[:, i, 2] 
                          + test_data_et[:, i, 4]*w[:, i, 4] 
                          + test_data_et[:, i, 6]*w[:, i, 6] 
                          + test_data_et[:, i, 8]*w[:, i, 8]).reshape(time_slot,)
            y[:, i, 0] = (test_data_et[:, i, 1]*w[:, i, 1] 
                          + test_data_et[:, i, 3]*w[:, i, 3] 
                          + test_data_et[:, i, 5]*w[:, i, 5] 
                          + test_data_et[:, i, 7]*w[:, i, 7] 
                          + test_data_et[:, i, 9]*w[:, i, 9]).reshape(time_slot,)
        elif i == 1:
            x[:, i, 0] = ((x[:, i-1, 0] + (speed[:, i, 0]*w[:, i, 12]).reshape(time_slot,))*w[:, i, 10]
                          + test_data_et[:, i, 0]*w[:, i, 0]
                          + test_data_et[:, i, 2]*w[:, i, 2]
                          + test_data_et[:, i, 4]*w[:, i, 4]
                          + test_data_et[:, i, 6]*w[:, i, 6]
                          + test_data_et[:, i, 8]*w[:, i, 8]).reshape(time_slot,)
            y[:, i, 0] = ((y[:, i-1, 0] + (speed[:, i, 0]*w[:, i, 13]).reshape(time_slot,))*w[:, i, 11]
                          + test_data_et[:, i, 1]*w[:, i, 1]
                          + test_data_et[:, i, 3]*w[:, i, 3]
                          + test_data_et[:, i, 5]*w[:, i, 5]
                          + test_data_et[:, i, 7]*w[:, i, 7]
                          + test_data_et[:, i, 9]*w[:, i, 9]).reshape(time_slot,) 
        else:
            x[:, i, 0] = ((x[:, i-1, 0] + (speed[:, i, 0]*w[:, i, 12]).reshape(time_slot,))*w[:, i, 10]
                          +test_data_et[:, i, 0]*w[:, i, 0]
                          +test_data_et[:, i, 2]*w[:, i, 2]
                          +test_data_et[:, i, 4]*w[:, i, 4]
                          +test_data_et[:, i, 6]*w[:, i, 6]
                          +test_data_et[:, i, 8]*w[:, i, 8]).reshape(time_slot,) 
            y[:, i, 0] = ((y[:, i-1, 0] + (speed[:, i, 0]*w[:, i, 13]).reshape(time_slot,))*w[:, i, 11]
                          + test_data_et[:, i, 1]*w[:, i, 1]
                          + test_data_et[:, i, 3]*w[:, i, 3]
                          + test_data_et[:, i, 5]*w[:, i, 5]
                          + test_data_et[:, i, 7]*w[:, i, 7]
                          + test_data_et[:, i, 9]*w[:, i, 9]).reshape(time_slot,) 
            
    result_x = x[:, time_slot_steps-1, 0]
    result_y = y[:, time_slot_steps-1, 0]  
    
    std_deviation, mean, err = estimate_EW_LSTM_v1.E(test_c_x, test_c_y, result_x, result_y, 
                                          time_slot, time_slot_steps, environment)
    
    sio.savemat(f'{address2}\\EWLSTM_v2_t{time_slot_steps}_{environment}_test_data_et.mat', {"test_data_et": test_data_et})
    sio.savemat(f'{address2}\\EWLSTM_v2_t{time_slot_steps}_{environment}_x.mat', {"x": x})
    sio.savemat(f'{address2}\\EWLSTM_v2_t{time_slot_steps}_{environment}_y.mat', {"y": y})
    print("std_deviation:", std_deviation)
    print("mean:", mean)
    # print(result_x)
#%% 
    weight = w.reshape(int(time_slot_steps*time_slot),14)
    sio.savemat(f'{address2}\\EWLSTM_v2_t{time_slot_steps}_{environment}_weight.mat', {"weight": weight})
    open(f'{address2}\\EWLSTM_v2_t{time_slot_steps}_{environment}_weight.csv', 'w').close()
    with open(f'{address2}\\EWLSTM_v2_t{time_slot_steps}_{environment}_weight.csv',mode='a', newline='') as file:
        writer = csv.writer(file)
        for item in weight:
            writer.writerow(item)