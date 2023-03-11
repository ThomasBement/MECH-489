# ---------------------------------------- #
# data_analysis [Python File]
# Written By: Thomas Bement
# Created On: 2023-03-10
# ---------------------------------------- #

"""
IMPORTS
"""
import os
import math
import matplotlib.ticker

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tabulate import tabulate
from scipy.optimize import curve_fit

"""
CONSTANTS
"""
data_path = './DATA'
image_path = './IMG'
headers = np.array(['Time [s]', 'Ch 1 [C]', 'Ch 2 [C]', 'Ch 3 [C]', 
                    'Ch 4 [C]', 'Ch 5 [C]', 'Ch 6 [C]', 'Ch 7 [C]', 
                    'Ch 8 [C]', 'Ch 9 [C]', 'Ch 10 [C]', 'Ch 11 [C]', 
                    'Ch 12 [C]', 'Ch 13 [C]', 'Ch 14 [C]', 'Ch 15 [C]', 
                    'T_Jet [C]', 'Ch 16 [C]', 'Ch 17 [C]', 'Ch 18 [C]', 
                    'Ch 19 [C]', 'Ch 20 [C]', 'Ch 21 [C]', 'Ch 22 [C]', 
                    'Ch 23 [C]', 'Ch 25 [C]', 'Ch 26 [C]', 'Ch 27 [C]', 
                    'Ch 28 [C]', 'Ch 29 [C]', 'Ch 30 [C]', 'Ch 31 [C]', 
                    'T_ambient [C]', 'T_Heat Flux [C]', 'Heat Flux [W/m^2]', 
                    'P_heater [W]'])
temp_mapping = [['Ch 1 [C]','Ch 2 [C]','Ch 3 [C]','Ch 4 [C]','Ch 5 [C]','Ch 6 [C]','Ch 7 [C]',
                'Ch 8 [C]','Ch 9 [C]','Ch 10 [C]','Ch 11 [C]','Ch 12 [C]','Ch 13 [C]','Ch 14 [C]',
                'Ch 15 [C]','Ch 16 [C]','Ch 17 [C]','Ch 18 [C]','Ch 19 [C]','Ch 20 [C]','Ch 21 [C]',
                'Ch 22 [C]','Ch 23 [C]','Ch 25 [C]','Ch 26 [C]','Ch 27 [C]','Ch 28 [C]','Ch 29 [C]',
                'Ch 30 [C]','Ch 31 [C]'],[0,0.5,1,1.5,2,2.5,3,3.5,4,0.75,1.25,1.75,2.25,2.75,3.25,3.75,
                0.5,1,1.5,2,2.5,3,3.5,0.75,1.25,1.75,2.25,2.75,3.25,3.75]]

t_inf = 1*3600
ID = 0.156

"""
FUNCTIONS
"""
def get_file(search_path): 
    files = os.listdir(search_path) 
    print('Files in %s:' %(search_path)) 
    data = [] 
    for i in range(len(files)): 
        data.append([files[i], '%i' %(i)]) 
    print(tabulate(data, headers=['File Name', 'Index']), '\n') 
    idx = input('Enter index of file name to be used: ')
    idx_lis = [int(i) for i in idx.split(',') if i.isdigit()]
    file_lis = [files[i] for i in idx_lis]
    return file_lis 

def read_data(file_paths, data_path, delim=','):
    ans = {}
    for file_path in file_paths:
        read_path = '%s/%s' %(data_path, file_path)
        data = pd.read_csv(read_path)
        for header in data:
            ans[header] = np.array(data[header])
    return ans

def t_cool(t, a, b, c, d):
    return a/(t-b)**c + d

def fit_dat(time, temp, time_max):
    idx = np.argmax(temp)
    X = time[idx:]
    Y = temp[idx:]

    p0 = [100, time[idx], 1, Y[-1]]
    popt, _ = curve_fit(t_cool, X, Y, p0=p0)
    x_fit = np.linspace(X[0], time_max, 2**9)
    y_fit = t_cool(x_fit, *popt)

    return [x_fit, y_fit, popt]

def h_conv(heat_flux, temp_surf, temp_jet):
    return np.abs(heat_flux/temp_jet-temp_surf)

"""
MAIN
"""
read_files = get_file(data_path)
data = read_data(read_files, data_path)

time = data['Time [s]']
flux = data['Heat Flux [W/m^2]'][-1]
TJ = data['T_Jet [C]'][-1]
R = []
HCONV = []
TS = []
for key in data:
    if 'Ch'in key:
        idx = 0
        temp = data[key]
        [time_fit, temp_fit, param] = fit_dat(time, temp, t_inf)
        TS.append(temp_fit[-1])
        for i in range(len(temp_mapping[0])):
            if (key == temp_mapping[0][i]):
                R.append(temp_mapping[1][i])
                HCONV.append(h_conv(flux, temp_fit[-1], TJ))
                break
plt.scatter(np.array(R)/np.array(ID), np.array(HCONV))
plt.legend()
plt.show()