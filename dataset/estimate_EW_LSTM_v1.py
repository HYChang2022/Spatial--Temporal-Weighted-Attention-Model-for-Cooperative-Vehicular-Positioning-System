#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 00:34:13 2023

@author: rex
"""
import statistics
import numpy as np    
import matplotlib.pyplot as plt
import scipy.io as sio

def E(test_c_x, test_c_y, result_x, result_y, t, time_slot, environment):
    address = f"D:\\EW-LSTM\\GPS_estimate\\Code\\v2\\result\\{environment}\\LSTM_Weight_2\\velocity_400"
    test_c = np.column_stack((test_c_x, test_c_y))
    result = np.column_stack((result_x, result_y))
    g = np.linalg.norm(test_c - result, axis=1)
    sorted_data = np.sort(g)
    cumulative_frequency = np.cumsum(sorted_data)
    cdf = (cumulative_frequency) / np.sum(g)
    std_deviation = statistics.stdev(g)
    mean = statistics.mean(g)
    # 將資料存入陣列
    data_to_save = np.array([sorted_data, cdf])
    np.save(f'{address}\\LSTM_Weight_2_cdf_data_{time_slot}.npy', data_to_save)
    
    variable_name = 'sorted_data'
    variable_name_2 = 'cdf'
    sio.savemat(f'{address}\\LSTM_Weight_2_cdf_data_{time_slot}.mat', {variable_name: sorted_data, variable_name_2 : cdf})
    
    plt.plot(sorted_data, cdf, label=
             f'Time slot: {time_slot}\n'
             #f'Data number: {t}\n' 
             f'std: {std_deviation:.6f}(m)\n'
             f'mean: {mean:.6f}(m)')
    plt.xlabel('error(m)')
    plt.ylabel('Cumulative Probability')
    plt.title('LSTM Weight Cumulative Distribution Function')
    plt.legend()
    plt.savefig(f'{address}\\LSTM_Weight_2_cdf_data_{time_slot}.png')
    plt.show()
    return std_deviation, mean, g

