# ---------------------------------------- #
# Curve_Fit [Python File]
# Written By: Thomas Bement
# Created On: 2023-02-15
# ---------------------------------------- #

import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

"""
FUNCTIONS
"""
# Color maps for plotting
def get_cmap(n, name='plasma'):
    return plt.cm.get_cmap(name, n)

# Coefficent of determination for curve fit
def coeff_determ(y, y_fit):
    y = np.array(y)
    y_fit = np.array(y_fit)
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - (ss_res / ss_tot)

# Model for curve fitting
def load_model(press, m, b):
    return (m*press) + b

# Fits based on model function
def model_fit(xDat, yDat, ySig):
    p0 = [1, 1]
    popt, pcov = curve_fit(load_model, xDat, yDat, p0=p0, sigma=ySig, absolute_sigma=True)
    x_fit = np.linspace(min(xDat), max(xDat), 256)
    y_fit = load_model(x_fit, *popt)
    m, b = popt
    return [x_fit, y_fit, m, b]

"""
MAIN
"""
allDat = {'Force': np.array([1.11,3.07,5.04,7.00,8.96,13.87,18.77,28.58,38.39,48.20,58.01,67.82])
        , 'Lift': np.array([1.305,3.244,5.218,7.17,9.162,14.144,19.141,29.122,39.087,49.007,58.998,68.905])
        , 'Lift Sig': np.array([0.026,0.026,0.026,0.024,0.03,0.025,0.043,0.022,0.029,0.028,0.026,0.027])
        , 'Drag': np.array([1.206,3.152,5.117,7.066,9.034,13.929,18.837,28.677,38.527,48.368,58.171,68.031])
        , 'Drag Sig': np.array([0.02,0.021,0.026,0.019,0.02,0.015,0.022,0.023,0.019,0.022,0.02,0.026])}

cmap = get_cmap(5, 'plasma')

lift_fit = model_fit(allDat['Force'], allDat['Lift'], allDat['Lift Sig'])
lift_fit.append(coeff_determ(allDat['Lift'], load_model(allDat['Lift'], lift_fit[2], lift_fit[3])))

drag_fit = model_fit(allDat['Force'], allDat['Drag'], allDat['Drag Sig'])
drag_fit.append(coeff_determ(allDat['Drag'], load_model(allDat['Drag'], drag_fit[2], drag_fit[3])))

plt.errorbar(allDat['Force'], allDat['Lift'], allDat['Lift Sig'], color = cmap(1), alpha = 0.5, fmt='o', markersize=4, capsize=5, label='Measured Lift')
plt.errorbar(allDat['Force'], allDat['Drag'], allDat['Drag Sig'], color = cmap(2), alpha = 0.5, fmt='o', markersize=4, capsize=5, label='Measured Drag')
plt.plot(lift_fit[0], lift_fit[1], color = cmap(1), alpha = 0.5, label='Fit Lift: %.4e*x + %.4e, R^2: %.4f' %(lift_fit[2], lift_fit[3], lift_fit[4]))
plt.plot(drag_fit[0], drag_fit[1], color = cmap(2), alpha = 0.5, label='Fit Drag: %.4e*x + %.4e, R^2: %.4f' %(drag_fit[2], drag_fit[3], drag_fit[4]))

plt.xlabel('Applied Force [N]')
plt.ylabel('Reading [N]')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2, fancybox=True, shadow=True)
plt.grid()

plt.savefig('.\IMG\Lift_Drag_Calibration.png', format='png', bbox_inches='tight')

print('LIFT DATA\n-----------------------------------')
print('Lift m: %.12f' %(lift_fit[2]))
print('Lift b: %.12f' %(lift_fit[3]))
print('DRAG DATA\n-----------------------------------')
print('Drag m: %.12f' %(drag_fit[2]))
print('Drag b: %.12f' %(drag_fit[3]))