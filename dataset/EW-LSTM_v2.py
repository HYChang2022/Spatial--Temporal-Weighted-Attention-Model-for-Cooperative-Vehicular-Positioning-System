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

import estimate_EW_LSTM_v2
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
import statistics 
import matplotlib.pyplot as plt
from keras import regularizers, optimizers
import scipy.io as sio

open(f'D:\\EW-LSTM\\GPS_estimate\\Code\\v3\\mean.csv', 'w').close()
number_epochs = 500
batch_size_n = 100
wk = 0.5
vehicle_number = 5
environment = '4.7_2.2'#'4.7_2.2'#'20.5_8.5'#'14.8_7'
address = f'D:\\EW-LSTM\\DataSets\\precision-gnss-main\\trajectories\\comb\\{environment}' 
#address = f'C:\\Users\\826BK2023\\Desktop\\DataSets\\precision-gnss-main\\trajectories\\trajectory_01\\easy\\{environment}'

#%% data_test_train.csv estimate_target_test_train
for time_slot_steps in range(5,6): 
    data_name = f'train_data_mt_{time_slot_steps}_{environment}_v5.csv'
    estimate_data_name = 'train_estimate_target_mt_{time_slot_steps}_{environment}_v5.csv'
    data_size = 0
    with open(f'{address}\\{data_name}', 'r') as f:
        reader = csv.reader(f)
        data = [list(map(float, row)) for row in reader]
    # 將Python list轉換為NumPy array
    arr1 = np.array(data)
    data_size = arr1.shape[0]
    # 五輛車估計的目標車輛位置 estimate_target_test   
    with open(f'{address}\\train_estimate_target_mt_{time_slot_steps}_{environment}_v{vehicle_number}.csv', newline='') as file:
        reader = csv.reader(file)
        data = [list(map(float, row)) for row in reader]
    arr2 = np.array(data)
    for i in range(5):
        arr2[:, 2*i] = (arr2[:, 2*i]*(1-wk)+arr2[:, 0]*(wk))
        arr2[:, 2*i+1] = (arr2[:, 2*i+1]*(1-wk)+arr2[:, 1]*(wk))
    
    train_coordinate_x = arr1[:, 0].reshape((data_size, 1))
    train_coordinate_y = arr1[:, 1].reshape((data_size, 1))
    train_coordinate = np.concatenate((train_coordinate_x, train_coordinate_y), axis=1)
    speed = arr1[:, 32].reshape(int(data_size/time_slot_steps), time_slot_steps, 1)
    
    train_data = np.concatenate((train_coordinate, arr2), axis=1)
    validation_data = train_data.reshape(int(data_size/time_slot_steps), time_slot_steps, 12)
    validation_data = np.concatenate((validation_data, speed), axis=2)
    
    train_et = arr1[:, 10:33].reshape(int(data_size/time_slot_steps), time_slot_steps, 23)
    
    #%% # 定義model
    
    @tf.function
    def custom_loss(y_true, y_pred):
       
        y_true=tf.reshape(y_true, (batch_size_n, time_slot_steps, 13))
        loss = tf.zeros([])
        gps_pred = tf.TensorArray(tf.float32, size=time_slot_steps)
        for i in range(time_slot_steps):
            if i == 0:
                gps_pred_x_t0 = tf.expand_dims(y_true[:, 0, 2]*y_pred[:, 0, 0] + 
                                               y_true[:, 0, 4]*y_pred[:, 0, 2] + 
                                               y_true[:, 0, 6]*y_pred[:, 0, 4] +
                                               y_true[:, 0, 8]*y_pred[:, 0, 6] + 
                                               y_true[:, 0, 10]*y_pred[:, 0, 8], axis=1)
                
                gps_pred_y_t0 = tf.expand_dims(y_true[:, 0, 3]*y_pred[:, 0, 1] + 
                                               y_true[:, 0, 5]*y_pred[:, 0, 3] + 
                                               y_true[:, 0, 7]*y_pred[:, 0, 5] +
                                               y_true[:, 0, 9]*y_pred[:, 0, 7] + 
                                               y_true[:, 0, 11]*y_pred[:, 0, 9], axis=1)
                
                gps_pred = gps_pred.write(0, tf.concat([gps_pred_x_t0, gps_pred_y_t0], axis=1))  
            #"""    
            elif i == 1:
                
                gps_pred_x = tf.expand_dims((gps_pred_x_t0[:, 0] + y_true[:, i, 12]*y_pred[:, i, 12])*y_pred[:, i, 10] +
                                            y_true[:, i, 2]*y_pred[:, i, 0] + 
                                            y_true[:, i, 4]*y_pred[:, i, 2] + 
                                            y_true[:, i, 6]*y_pred[:, i, 4] +
                                            y_true[:, i, 8]*y_pred[:, i, 6] + 
                                            y_true[:, i, 10]*y_pred[:, i, 8], axis=1)
                
                gps_pred_y = tf.expand_dims((gps_pred_y_t0[:, 0]+y_true[:, i, 12]*y_pred[:, i, 13])*y_pred[:, i, 11]+
                                            y_true[:, i, 3]*y_pred[:, i, 1] + 
                                            y_true[:, i, 5]*y_pred[:, i, 3] + 
                                            y_true[:, i, 7]*y_pred[:, i, 5] +
                                            y_true[:, i, 9]*y_pred[:, i, 7] + 
                                            y_true[:, i, 11]*y_pred[:, i, 9], axis=1)
                gps_pred = gps_pred.write(i, tf.concat([gps_pred_x, gps_pred_y], axis=1))
            else:
                gps_pred_x = tf.expand_dims((gps_pred.read(i-1)[:, 0]+y_true[:, i, 12]*y_pred[:, i, 12])*y_pred[:, i, 10] + 
                                            y_true[:, i, 2]*y_pred[:, i, 0] + 
                                            y_true[:, i, 4]*y_pred[:, i, 2] + 
                                            y_true[:, i, 6]*y_pred[:, i, 4] +
                                            y_true[:, i, 8]*y_pred[:, i, 6] + 
                                            y_true[:, i, 10]*y_pred[:, i, 8], axis=1)
                
                gps_pred_y = tf.expand_dims((gps_pred.read(i-1)[:, 1]+y_true[:, i, 12]*y_pred[:, i, 13])*y_pred[:, i, 11] + 
                                            y_true[:, i, 3]*y_pred[:, i, 1] + 
                                            y_true[:, i, 5]*y_pred[:, i, 3] + 
                                            y_true[:, i, 7]*y_pred[:, i, 5] +
                                            y_true[:, i, 9]*y_pred[:, i, 7] + 
                                            y_true[:, i, 11]*y_pred[:, i, 9], axis=1)
                
                gps_pred = gps_pred.write(i, tf.concat([gps_pred_x, gps_pred_y], axis=1))
            #""" 
            loss += K.sqrt(K.square(gps_pred.read(i)[:, 0] - y_true[:, i, 0]) 
                           + K.square(gps_pred.read(i)[:, 1] - y_true[:, i, 1])) 
        return loss
                                              
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(100, input_shape=(time_slot_steps, 23), return_sequences=True),
        #tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.LSTM(64, return_sequences=True),
        #tf.keras.layers.LSTM(32, return_sequences=True),
        #tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(14, activation='linear')
    ])
    
    tf.keras.utils.plot_model(
        model,
        to_file=f'model_plot_{time_slot_steps}.png',
        show_shapes=True,
        show_layer_names=True,
        rankdir='TB'  # 可选，用于设置图的方向（上下或左右）
    )
        
    tf.keras.utils.get_custom_objects()['custom_loss'] = custom_loss
    
    adam = optimizers.Adam(lr=0.0001)
    
    model.compile(optimizer='adam', loss='custom_loss')
    
    
    checkpoint = ModelCheckpoint(f'D:\\EW-LSTM\\GPS_estimate\\Code\\v3\\result\\{environment}\\LSTM_Weight_2\\model_weights_{time_slot_steps}.h5', 
                                 save_best_only=True,  # 只保存最好的模型
                                 mode='min')
    
    gps = model.fit(train_et, validation_data, 
                    epochs=number_epochs, batch_size=batch_size_n, 
                    validation_split=0.2, callbacks=[checkpoint])
    
    train_loss = gps.history['loss']
    val_loss = gps.history['val_loss']
    #%%
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_loss, 'ro', label='Training Loss')
    plt.plot(epochs, val_loss, 'b*', label='Validation Loss')
    plt.title(f'Time Slot {time_slot_steps} Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    #plt.xlim(number_epochs*0, number_epochs)
    #plt.ylim(5, 50)
    plt.legend()
    plt.show()
    sio.savemat(f'D:\\EW-LSTM\\weight\\EWLSTM_v3_t{time_slot_steps}_{environment}_train_loss.mat', {"train_loss": train_loss})
    sio.savemat(f'D:\\EW-LSTM\\weight\\EWLSTM_v3_t{time_slot_steps}_{environment}_val_loss.mat', {"val_loss": val_loss})
    #%% 
    result_x = []
    result_y = []
    
    
    with open(f'{address}\\test_data_mt_{time_slot_steps}_{environment}_v{vehicle_number}.csv', 'r') as f:
        reader = csv.reader(f)
        err1 = [list(map(float, row)) for row in reader]
    err1 = np.array(err1)
    data_size = err1.shape[0]
    with open(f'{address}\\test_estimate_target_mt_{time_slot_steps}_{environment}_v{vehicle_number}.csv', newline='') as f:
        reader = csv.reader(f)
        err2 = [list(map(float, row)) for row in reader]
    err2 = np.array(err2)
    
    for i in range(5):
        err2[:, 2*i] = (err2[:, 2*i]*(wk)+err2[:, 0]*(1-wk))
        err2[:, 2*i+1] = (err2[:, 2*i+1]*(wk)+err2[:, 1]*(1-wk))
    
    speed = err1[:, 32].reshape(int(data_size/time_slot_steps), time_slot_steps, 1)
    test_c_x = err1[(time_slot_steps-1)::time_slot_steps, 0]
    test_c_y = err1[(time_slot_steps-1)::time_slot_steps, 1]
    
    test_data = err1[:, 10:33].reshape(int(data_size/time_slot_steps), time_slot_steps, 23)
    test_data_et = err2.reshape(int(data_size/time_slot_steps), time_slot_steps, 10)
    x = np.zeros((int(data_size/time_slot_steps), time_slot_steps, 1))
    y = np.zeros((int(data_size/time_slot_steps), time_slot_steps, 1))  
    #%%                 
    w = model.predict(test_data)
    for i in range(time_slot_steps):
        if i == 0:
            x[:, i, 0] = (test_data_et[:, i, 0]*w[:, i, 0] 
                          + test_data_et[:, i, 2]*w[:, i, 2] 
                          + test_data_et[:, i, 4]*w[:, i, 4] 
                          + test_data_et[:, i, 6]*w[:, i, 6] 
                          + test_data_et[:, i, 8]*w[:, i, 8]
                          + w[:, i, 10]*w[:, i, 11]).reshape(int(data_size/time_slot_steps),)
            
            y[:, i, 0] = (test_data_et[:, i, 1]*w[:, i, 1] 
                          + test_data_et[:, i, 3]*w[:, i, 3] 
                          + test_data_et[:, i, 5]*w[:, i, 5] 
                          + test_data_et[:, i, 7]*w[:, i, 7] 
                          + test_data_et[:, i, 9]*w[:, i, 9]
                          + w[:, i, 12]*w[:, i, 13]).reshape(int(data_size/time_slot_steps),)
        else:
            x[:, i, 0] = ((x[:, i-1, 0] + (speed[:, i, 0]*w[:, i, 12]).reshape(int(data_size/time_slot_steps),))*w[:, i, 10] + 
                          test_data_et[:, i, 0]*w[:, i, 0]
                          + test_data_et[:, i, 2]*w[:, i, 2]
                          + test_data_et[:, i, 4]*w[:, i, 4]
                          + test_data_et[:, i, 6]*w[:, i, 6]
                          + test_data_et[:, i, 8]*w[:, i, 8]
                          + w[:, i, 10]*w[:, i, 11]).reshape(int(data_size/time_slot_steps),)
            
            y[:, i, 0] = ((y[:, i-1, 0] + (speed[:, i, 0]*w[:, i, 13]).reshape(int(data_size/time_slot_steps),))*w[:, i, 11] + 
                          test_data_et[:, i, 1]*w[:, i, 1]
                          + test_data_et[:, i, 3]*w[:, i, 3]
                          + test_data_et[:, i, 5]*w[:, i, 5]
                          + test_data_et[:, i, 7]*w[:, i, 7]
                          + test_data_et[:, i, 9]*w[:, i, 9]
                          + w[:, i, 12]*w[:, i, 13]).reshape(int(data_size/time_slot_steps),)
            
    result_x = x[:, time_slot_steps-1, 0]
    result_y = y[:, time_slot_steps-1, 0]      
    std_deviation, mean, err = estimate_EW_LSTM_v1.E(test_c_x, test_c_y, 
                                                      result_x, result_y, int(data_size/time_slot_steps), 
                                                      time_slot_steps, environment)
    address2 = f"D:\\EW-LSTM\\weight"
    sio.savemat(f'{address2}\\EWLSTM_v3_t{time_slot_steps}_{environment}_test_data_et.mat', {"test_data_et": test_data_et})
    sio.savemat(f'{address2}\\EWLSTM_v3_t{time_slot_steps}_{environment}_x.mat', {"x": x})
    sio.savemat(f'{address2}\\EWLSTM_v3_t{time_slot_steps}_{environment}_y.mat', {"y": y})
    print("std:", std_deviation)
    print("mean:", mean)
    # print(result_x)
    #%%
    weight = w.reshape(data_size,14)
    
    sio.savemat(f'{address2}\\EWLSTM_v3_t{time_slot_steps}_{environment}_weight.mat', {"weight": weight})
    open(f'{address2}\\EWLSTM_v3_t{time_slot_steps}_{environment}_weight.csv', 'w').close()
    with open(f'{address2}\\EWLSTM_v3_t{time_slot_steps}_{environment}_weight.csv',mode='a', newline='') as file:
        writer = csv.writer(file)
        for item in weight:
            writer.writerow(item)