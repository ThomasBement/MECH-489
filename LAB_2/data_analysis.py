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
t_inf = 1*3600
ID = 0.156*0.0254

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

def read_data(file_paths, data_path, delim=',', single=True):
    ans = {}
    if single:
        for file_path in file_paths:
            read_path = '%s/%s' %(data_path, file_path)
            data = pd.read_csv(read_path)
            for header in data:
                ans[header] = np.array(data[header])
    else:
        for file_path in file_paths:
            ans[file_path] = {}
            read_path = '%s/%s' %(data_path, file_path)
            data = pd.read_csv(read_path)
            for header in data:
                ans[file_path][header] = np.array(data[header])
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
    return np.abs(heat_flux/(temp_jet-temp_surf))

def q_heater(power_heater): # power [W]
    area = 0.048258 # m^2
    return power_heater/area

def q_rad(temp_surf):
    emissivity = 0.16
    sigma = 5.670e-8 # [W/m^2/K^4] Stefan-Boltzmann const
    return emissivity*sigma*temp_surf**4

def nuss(h, d, k):
    return h*d/k

def reynolds(Q, d, v):
    return (4*Q)/(np.pi*d*v)

def flow(x, a=-2.27E-6, b=8.695E-4, c=-0.11726, d=7.4009, e=189.6027):
    return a*x**5 +b*x**4 + c*x**3 + d*x**2 + e*x

def mart_AR(d, r):
    return (d**2)/(4*r**2)

def mart_G(HD, AR):
    return 2*np.sqrt(AR)*((1-2.2*np.sqrt(AR))/(1+0.2*(HD-6)*np.sqrt(AR)))

def mart_NU(G, RE):
    return G*(2*np.sqrt(RE)*np.sqrt(1+0.005*RE**0.55))

def part_a(data, name):
    TC_MAP = read_data(['TC_MAP.csv'], './DATA/OTHER')
    TC_303 = read_data(['303_TC.csv'], './DATA/OTHER')
    VISC_MAP = read_data(['VISC_MAP.csv'], './DATA/OTHER')
    
    time = data['Time [s]']
    flux = data['Heat Flux [W/m^2]'][-1]
    TJ = data['T_Jet [C]'][-1]
    kin_visc = np.interp(TJ, VISC_MAP['Temperature [C]'], VISC_MAP['Kinematic Viscosity [m2/s *10-6]'])*(1E-6)
    Q = flow(float(name.split('.')[0].split('_')[0].split('Q')[-1]))*(1E-6)*(1/60)
    HD = float(name.split('HR')[-1].split('.')[0].replace('d', '.'))
    RE = reynolds(Q, ID, kin_visc)

    RD = []
    TS = []
    HCONV = []  
    NU = []
    for key in data:
        if 'Ch'in key:
            idx = 0
            temp = data[key]
            [time_fit, temp_fit, param] = fit_dat(time, temp, t_inf)
            TS.append(temp_fit[-1])
            for i in range(len(TC_MAP['Channel'])):
                if (key == TC_MAP['Channel'][i]):
                    RD.append(TC_MAP['R/D'][i])
                    HCONV.append(h_conv(flux, temp_fit[-1], TJ))
                    NU.append(nuss(HCONV[-1], ID, np.interp(TS[-1], TC_303['Temperature [C]'], TC_303['Thermal Conductivity [W/m-K]'])))
                    break
    plt.title('Local Nu vs. Radial Position for Re: %.0f and H/D: %.1s' %(RE, HD))
    plt.scatter(RD, NU)
    plt.xlabel('Radial Position Normalized by Jet Diameter [N.a.]')
    plt.ylabel('Local Nusselt Number [N.a.]')
    plt.savefig('%s/%s.png' %(image_path, name.split('.')[0]), format='png', bbox_inches='tight')
    plt.show()

def part_f():
    RE = [12000, 20000]
    HD = [2.5, 10]
    read_files = get_file(data_path)
    data = read_data(read_files, data_path, single=False)
    for key in data:
        print(key)


"""
MAIN
"""
read_files = get_file(data_path)
data = read_data(read_files, data_path)

print('PART A:\n\n')
part_a(data, read_files[0])

quit()
print('PART F:\n\n')
part_f()