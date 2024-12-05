# -*- coding: utf-8 -*-
"""
Created on Fri May  5 11:41:59 2023
正確位置, gps位置, (Radar距離, 角度), RSSI
目標車輛, 第一台車, 第二台車, 第三台車, 第四台車
每n個row為一組time slot
(這是只有目標車輛有速度關係)
@author: 826BK2023
"""
import random
import numpy as np
import math
import csv

K = 100000
vehicle_nunber = 5
#%%
## N(4.7, 2.2), N(14.8, 7), N(20.5, 8.5)
loc=4.7
scale=2.2
environment = f'{loc}_{scale}'
address = f'D:\\EW-LSTM\\GPS_estimate\\Code\\v2\DATA\\{environment}_w_error\\sensor_error_4x'

#%%
for time_slot_n in range(5,6):
    name = 'train_'
    open(f'{address}\\{name}data_mt_{time_slot_n}_{environment}_v{vehicle_nunber}.csv', 'w').close()
    for a in range(K):
        gps_coordinate = []
        radar_error = []
        points = []
        speed = np.zeros((time_slot_n, 1))
        coordinate_0 = [random.uniform(30, 300), random.uniform(30, 300)]
        coordinate = np.zeros((time_slot_n, 2))
        coordinate[0] = coordinate_0
        #生成目標車輛座標
        for i in range(1, time_slot_n):
            speed[i] = random.uniform(100/9, 50/3)*0.5
            s_x = random.uniform(0,1)
            s_y = (1-s_x**2)**0.5
            #print(s_x**2+s_y**2)
            coordinate[i] = [coordinate[i-1, 0] +speed[i]*s_x, coordinate[i-1, 1]+speed[i]*s_y]#這裡改speed x y 
        test_data = []
        test_data = coordinate
        # 生成其餘四台車座標
        for i in range(4):
            radius = np.random.uniform(1, 10, (time_slot_n, 1))
            angle = np.random.uniform(0, 2 * math.pi, (time_slot_n, 1))
            else_vehicle_x = coordinate[:, 0].reshape((time_slot_n, 1)) + radius * np.cos(angle)
            else_vehicle_y = coordinate[:, 1].reshape((time_slot_n, 1)) + radius * np.sin(angle)
            else_vehicle = np.concatenate((else_vehicle_x, else_vehicle_y), axis=1)
            test_data = np.concatenate((test_data, else_vehicle), axis=1)
        
        # GPS error
        for i in range(5):
            GPS_error_free_space = np.random.normal(loc=loc, scale=scale, size=(time_slot_n, 1)) #environment = '4.7_2.2'
            #radius = np.random.uniform(0, GPS_error_free_space, (time_slot_n, 1))
            angle = np.random.uniform(0, 2 * math.pi, (time_slot_n, 1))
            
            gps_error_x = test_data[:, 0+i*2].reshape((time_slot_n, 1)) + GPS_error_free_space * np.cos(angle)
            gps_error_y = test_data[:, 1+i*2].reshape((time_slot_n, 1)) + GPS_error_free_space * np.sin(angle)
            
            gps_error = np.concatenate((gps_error_x, gps_error_y), axis=1)
            test_data = np.concatenate((test_data, gps_error), axis=1)
       
        # 產radar error   
        for i in range(5):
            d = np.sqrt((test_data[:, 0] - test_data[:, 0+i*2])**2 + (test_data[:, 1] - test_data[:, 1+i*2])**2)
            radar_distance_error = (np.random.normal(loc=0, scale=(((0.025*d*4)**2)))+d).reshape((time_slot_n, 1))
            
            angle_diff = np.arctan2(test_data[:, 1+i*2] - test_data[:, 1], test_data[:, 0+i*2] - test_data[:, 0])
            angle_diff_degrees = np.degrees(angle_diff).reshape((time_slot_n, 1))
            radar_phase_error = np.random.normal(loc=0, scale=2**2, size=(time_slot_n, 1)) + angle_diff_degrees
            radar_error = np.concatenate((radar_distance_error, radar_phase_error), axis=1)
            test_data = np.concatenate((test_data, radar_error), axis=1)
      
        # RSSI
        for i in range(5):
            d = np.sqrt((test_data[:, 0] - test_data[:, 0+i*2])**2 + (test_data[:, 1] - test_data[:, 1+i*2])**2)
            rho_0 = -34
            alpha = 2.1
            rssi_err = (np.random.normal(loc=5.5, scale=1, size=(time_slot_n, 1)))
            rssi = rho_0-10*alpha-(np.log(d.reshape((time_slot_n, 1))+rssi_err))/np.log(10)
            test_data = np.concatenate((test_data, rssi), axis=1)
        speed = np.zeros((time_slot_n, 1))+speed
        test_data = np.concatenate((test_data, speed), axis=1)
        
        with open(f'{address}\\{name}data_mt_{time_slot_n}_{environment}_v{vehicle_nunber}.csv',mode='a', newline='') as file:
            writer = csv.writer(file)
            for item in test_data:
                writer.writerow(item)
      
            
    #%% estimate_target
    open(f'{address}\\{name}estimate_target_mt_{time_slot_n}_{environment}_v{vehicle_nunber}.csv', 'w').close()
    with open(f'{address}\\{name}data_mt_{time_slot_n}_{environment}_v{vehicle_nunber}.csv', newline='') as file:
        reader = csv.reader(file)
        data = [list(map(float, row)) for row in reader]
    arr = np.array(data)
    cout = arr.shape[0]
    estimate_gps_target_x = (arr[:, 10] - arr[:, 20]*np.cos(arr[:, 21])).reshape((cout, 1))
    estimate_gps_target_y = (arr[:, 11] - arr[:, 20]*np.sin(arr[:, 21])).reshape((cout, 1))
    estimate_gps_target = np.concatenate((estimate_gps_target_x, estimate_gps_target_y), axis=1)
    for i in range(4):   #i為第幾台車
        x = (arr[:, 12+2*i] - arr[:, 22+2*i]*np.cos(arr[:, 23+2*i])).reshape((cout, 1))
        y = (arr[:, 13+2*i] - arr[:, 22+2*i]*np.sin(arr[:, 23+2*i])).reshape((cout, 1))
        estimate_gps = np.concatenate((x, y), axis=1)
        estimate_gps_target = np.concatenate((estimate_gps_target, estimate_gps), axis=1) 
    with open(f'{address}\\{name}estimate_target_mt_{time_slot_n}_{environment}_v{vehicle_nunber}.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        for item in estimate_gps_target:
            writer.writerow(item)
    print('已寫入csv')
    #%% 產H矩陣
    open(f'{address}\\{name}data_mt_{time_slot_n}_{environment}_vH{vehicle_nunber}.csv', 'w').close()
    H = np.zeros((K,vehicle_nunber,(2+4+4+4)*time_slot_n+1))#np.zeros((K,vehicle_nunber,(2+4+4+4)*time_slot_n+1))
    # 多產一組radar_error,RSSI  
    radar_error_c = np.zeros((K*time_slot_n,(vehicle_nunber-1)*3))
    for i in range(1,5):
        d = np.sqrt((arr[:, 0] - arr[:, 0+i*2])**2 + (arr[:, 1] - arr[:, 1+i*2])**2)
        radar_d_error = (np.random.normal(loc=0, scale=(((0.025*d*4)**2)))+d).reshape((K*time_slot_n,))
            
        angle_dif = np.arctan2(arr[:,1+i*2] - arr[:,1], arr[:,0+i*2] - arr[:, 0])
        angle_dif_degrees = np.degrees(angle_dif).reshape((K*time_slot_n,))
        radar_p_error = np.random.normal(loc=0, scale=2**2, size=(K*time_slot_n,)) + angle_dif_degrees
        
        rssi_err_c = (np.random.normal(loc=5.5, scale=1, size=(K*time_slot_n,)))
        rssi_c = rho_0-10*alpha-(np.log(d.reshape((K*time_slot_n,))+rssi_err_c))/np.log(10)
        radar_error_c[:,i-1] = radar_d_error
        radar_error_c[:,3+i] = radar_p_error
        radar_error_c[:,7+i] = rssi_c
    #
    for k in range(K):
        for t in range(time_slot_n):
            for i in range(vehicle_nunber):
                H[k,i,0+(14*t)] = arr[0+t+(k*time_slot_n),(vehicle_nunber*2  )+(2*i)]
                H[k,i,1+(14*t)] = arr[0+t+(k*time_slot_n),(vehicle_nunber*2+1)+(2*i)] #車輛座標
            for i in range(vehicle_nunber-1):
                H[k,0,2+i+(14*t)] = arr[0+t+(k*time_slot_n),(vehicle_nunber*4+2)+(2*i)]
                H[k,0,6+i+(14*t)] = arr[0+t+(k*time_slot_n),(vehicle_nunber*4+3)+(2*i)] #目標車輛收集訊息(Radar)
                #對角線
                H[k,1+i,2+i+(14*t)] = radar_error_c[0+t+(k*time_slot_n),0+i]
                H[k,1+i,6+i+(14*t)] = radar_error_c[0+t+(k*time_slot_n),4+i]
                H[k,1+i,10+i+(14*t)] = radar_error_c[0+t+(k*time_slot_n),8+i] #(周圍車輛收集訊息)
            H[k,0,10+(14*t):14+(14*t)] = arr[0+t+(k*time_slot_n),(vehicle_nunber*6+1):(vehicle_nunber*7)] #RSSI
    
        H[k,0:time_slot_n,14*time_slot_n] = arr[0 + (k*time_slot_n) : time_slot_n + (k*time_slot_n),35]
    H = H.reshape((K*vehicle_nunber,(2+4+4+4)*time_slot_n+1)) #H.reshape((K*vehicle_nunber,(2+4+4+4)*time_slot_n+1))
    with open(f'{address}\\{name}data_mt_{time_slot_n}_{environment}_vH{vehicle_nunber}.csv',mode='a', newline='') as file:
        writer = csv.writer(file)
        for item in H:
            writer.writerow(item)
    print('已寫入H.csv')
    
    name = 'test_'
    #%%
    open(f'{address}\\{name}data_mt_{time_slot_n}_{environment}_v{vehicle_nunber}.csv', 'w').close()
    for a in range(K):
        gps_coordinate = []
        radar_error = []
        points = []
        speed = np.zeros((time_slot_n, 1))
        coordinate_0 = [random.uniform(30, 300), random.uniform(30, 300)]
        coordinate = np.zeros((time_slot_n, 2))
        coordinate[0] = coordinate_0
        #生成目標車輛座標
        for i in range(1, time_slot_n):
            speed[i] = random.uniform(100/9, 50/3)*0.5
            s_x = random.uniform(0,1)
            s_y = (1-s_x**2)**0.5
            coordinate[i] = [coordinate[i-1, 0] +speed[i]*s_x, coordinate[i-1, 1]+speed[i]*s_y]#這裡改speed x y 
        test_data = []
        test_data = coordinate
        # 生成其餘四台車座標
        for i in range(4):
            radius = np.random.uniform(1, 10, (time_slot_n, 1))
            angle = np.random.uniform(0, 2 * math.pi, (time_slot_n, 1))
            else_vehicle_x = coordinate[:, 0].reshape((time_slot_n, 1)) + radius * np.cos(angle)
            else_vehicle_y = coordinate[:, 1].reshape((time_slot_n, 1)) + radius * np.sin(angle)
            else_vehicle = np.concatenate((else_vehicle_x, else_vehicle_y), axis=1)
            test_data = np.concatenate((test_data, else_vehicle), axis=1)
        
        # GPS error
        for i in range(5):
            GPS_error_free_space = np.random.normal(loc=loc, scale=scale, size=(time_slot_n, 1)) #environment = '4.7_2.2'
            #radius = np.random.uniform(0, GPS_error_free_space, (time_slot_n, 1))
            angle = np.random.uniform(0, 2 * math.pi, (time_slot_n, 1))
            
            gps_error_x = test_data[:, 0+i*2].reshape((time_slot_n, 1)) + GPS_error_free_space * np.cos(angle)
            gps_error_y = test_data[:, 1+i*2].reshape((time_slot_n, 1)) + GPS_error_free_space * np.sin(angle)
            
            gps_error = np.concatenate((gps_error_x, gps_error_y), axis=1)
            test_data = np.concatenate((test_data, gps_error), axis=1)
       
        # 產radar error   
        for i in range(5):
            d = np.sqrt((test_data[:, 0] - test_data[:, 0+i*2])**2 + (test_data[:, 1] - test_data[:, 1+i*2])**2)
            radar_distance_error = (np.random.normal(loc=0, scale=(((0.025*d*4)**2)))+d).reshape((time_slot_n, 1))
            
            angle_diff = np.arctan2(test_data[:, 1+i*2] - test_data[:, 1], test_data[:, 0+i*2] - test_data[:, 0])
            angle_diff_degrees = np.degrees(angle_diff).reshape((time_slot_n, 1))
            radar_phase_error = np.random.normal(loc=0, scale=2**2, size=(time_slot_n, 1)) + angle_diff_degrees
            radar_error = np.concatenate((radar_distance_error, radar_phase_error), axis=1)
            test_data = np.concatenate((test_data, radar_error), axis=1)
      
        # RSSI
        for i in range(5):
            d = np.sqrt((test_data[:, 0] - test_data[:, 0+i*2])**2 + (test_data[:, 1] - test_data[:, 1+i*2])**2)
            rho_0 = -34
            alpha = 2.1
            rssi_err = (np.random.normal(loc=5.5, scale=1, size=(time_slot_n, 1)))
            rssi = rho_0-10*alpha-(np.log(d.reshape((time_slot_n, 1))+rssi_err))/np.log(10)
            test_data = np.concatenate((test_data, rssi), axis=1)
        speed = np.zeros((time_slot_n, 1))+speed
        test_data = np.concatenate((test_data, speed), axis=1)
        
        with open(f'{address}\\{name}data_mt_{time_slot_n}_{environment}_v{vehicle_nunber}.csv',mode='a', newline='') as file:
            writer = csv.writer(file)
            for item in test_data:
                writer.writerow(item)
      
            
    #%% estimate_target
    open(f'{address}\\{name}estimate_target_mt_{time_slot_n}_{environment}_v{vehicle_nunber}.csv', 'w').close()
    with open(f'{address}\\{name}data_mt_{time_slot_n}_{environment}_v{vehicle_nunber}.csv', newline='') as file:
        reader = csv.reader(file)
        data = [list(map(float, row)) for row in reader]
    arr = np.array(data)
    cout = arr.shape[0]
    estimate_gps_target_x = (arr[:, 10] - arr[:, 20]*np.cos(arr[:, 21])).reshape((cout, 1))
    estimate_gps_target_y = (arr[:, 11] - arr[:, 20]*np.sin(arr[:, 21])).reshape((cout, 1))
    estimate_gps_target = np.concatenate((estimate_gps_target_x, estimate_gps_target_y), axis=1)
    for i in range(4):   #i為第幾台車
        x = (arr[:, 12+2*i] - arr[:, 22+2*i]*np.cos(arr[:, 23+2*i])).reshape((cout, 1))
        y = (arr[:, 13+2*i] - arr[:, 22+2*i]*np.sin(arr[:, 23+2*i])).reshape((cout, 1))
        estimate_gps = np.concatenate((x, y), axis=1)
        estimate_gps_target = np.concatenate((estimate_gps_target, estimate_gps), axis=1) 
    with open(f'{address}\\{name}estimate_target_mt_{time_slot_n}_{environment}_v{vehicle_nunber}.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        for item in estimate_gps_target:
            writer.writerow(item)
    print('已寫入csv')
    #%% 產H矩陣
    open(f'{address}\\{name}data_mt_{time_slot_n}_{environment}_vH{vehicle_nunber}.csv', 'w').close()
    H = np.zeros((K,vehicle_nunber,(2+4+4+4)*time_slot_n+1))#np.zeros((K,vehicle_nunber,(2+4+4+4)*time_slot_n+1))
    # 多產一組radar_error,RSSI  
    radar_error_c = np.zeros((K*time_slot_n,(vehicle_nunber-1)*3))
    for i in range(1,5):
        d = np.sqrt((arr[:, 0] - arr[:, 0+i*2])**2 + (arr[:, 1] - arr[:, 1+i*2])**2)
        radar_d_error = (np.random.normal(loc=0, scale=(((0.025*d*4)**2)))+d).reshape((K*time_slot_n,))
            
        angle_dif = np.arctan2(arr[:,1+i*2] - arr[:,1], arr[:,0+i*2] - arr[:, 0])
        angle_dif_degrees = np.degrees(angle_dif).reshape((K*time_slot_n,))
        radar_p_error = np.random.normal(loc=0, scale=2**2, size=(K*time_slot_n,)) + angle_dif_degrees
        
        rssi_err_c = (np.random.normal(loc=5.5, scale=1, size=(K*time_slot_n,)))
        rssi_c = rho_0-10*alpha-(np.log(d.reshape((K*time_slot_n,))+rssi_err_c))/np.log(10)
        radar_error_c[:,i-1] = radar_d_error
        radar_error_c[:,3+i] = radar_p_error
        radar_error_c[:,7+i] = rssi_c
    #
    for k in range(K):
        for t in range(time_slot_n):
            for i in range(vehicle_nunber):
                H[k,i,0+(14*t)] = arr[0+t+(k*time_slot_n),(vehicle_nunber*2  )+(2*i)]
                H[k,i,1+(14*t)] = arr[0+t+(k*time_slot_n),(vehicle_nunber*2+1)+(2*i)] #車輛座標
            for i in range(vehicle_nunber-1):
                H[k,0,2+i+(14*t)] = arr[0+t+(k*time_slot_n),(vehicle_nunber*4+2)+(2*i)]
                H[k,0,6+i+(14*t)] = arr[0+t+(k*time_slot_n),(vehicle_nunber*4+3)+(2*i)] #目標車輛收集訊息(Radar)
                #對角線
                H[k,1+i,2+i+(14*t)] = radar_error_c[0+t+(k*time_slot_n),0+i]
                H[k,1+i,6+i+(14*t)] = radar_error_c[0+t+(k*time_slot_n),4+i]
                H[k,1+i,10+i+(14*t)] = radar_error_c[0+t+(k*time_slot_n),8+i] #(周圍車輛收集訊息)
            H[k,0,10+(14*t):14+(14*t)] = arr[0+t+(k*time_slot_n),(vehicle_nunber*6+1):(vehicle_nunber*7)] #RSSI
    
        H[k,0:time_slot_n,14*time_slot_n] = arr[0+(k*time_slot_n):time_slot_n+(k*time_slot_n),35]
    H = H.reshape((K*vehicle_nunber,(2+4+4+4)*time_slot_n+1)) #H.reshape((K*vehicle_nunber,(2+4+4+4)*time_slot_n+1))
    with open(f'{address}\\{name}data_mt_{time_slot_n}_{environment}_vH{vehicle_nunber}.csv',mode='a', newline='') as file:
        writer = csv.writer(file)
        for item in H:
            writer.writerow(item)
    print('已寫入H.csv')