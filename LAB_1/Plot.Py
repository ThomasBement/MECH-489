# ---------------------------------------- #
# Plot [Python File]
# Written By: Thomas Bement
# Created On: 2023-02-15
# ---------------------------------------- #

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read in data_1
filName_1 = ".\DATA\Force_Attack.csv"
xVar_1 = "Angle of Attack"
data_1 = pd.read_csv(filName_1)

filName_2 = ".\DATA\Drag_Airspeed.csv"
xVar_2 = "Reynolds Number"
data_2 = pd.read_csv(filName_2)

filName_3 = ".\DATA\Static_Pressure.csv"
xVar_3 = "x"
data_3 = pd.read_csv(filName_3)

headers_1 = ['Angle of Attack', 'Cl Measured', 'Cl Theoretical', 'Cd Measured']
headers_2 = ['CD Measured', 'Reynolds Number']
headers_3 = ['x_top', 'Cp_top', 'x_bot', 'Cp_top']




# Make plots
plt.rc('axes', axisbelow=True)

# Lift coefficents
stall_min, stall_max = 0, 0
cl_max = -100
for i in range(len(data_1['Cl Measured'])):
    if (data_1['Cl Measured'][i] > cl_max):
        cl_max = data_1['Cl Measured'][i]
        stall_min = data_1[xVar_1][i]
        stall_max = data_1[xVar_1][i+1]

ymin = min([min(data_1['Cl Measured']), min(data_1['Cl Theoretical'])])
ymax = max([max(data_1['Cl Measured']), max(data_1['Cl Theoretical'])])

plt.vlines(stall_min, ymin, ymax, colors='r', linestyles='solid')
plt.vlines(stall_max, ymin, ymax, colors='r', linestyles='solid')
plt.fill([stall_min, stall_max, stall_max, stall_min], [ymin, ymin, ymax, ymax], color='r', alpha=0.5, label='Begins Stall')

plt.scatter([8], [1.0925343766712312], color = '#842ac9', label='Cl from Integrated Cp')
plt.plot(data_1[xVar_1], data_1['Cl Measured'], color='#0476d4', marker='.', label='Cl Measured')
plt.plot(data_1[xVar_1], data_1['Cl Theoretical'], color='#26bf96', marker='.', label='Cl Theoretical')


plt.xlabel('Angle of Attack [deg]')
plt.ylabel('Lift Coefficient')

plt.grid(b=True, which='both', axis='both')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2, fancybox=True, shadow=True)

plt.savefig('.\IMG\CL_Meas_Ther.png', format='png', bbox_inches='tight')
plt.show()
plt.close()

# Lift and drag
stall_min, stall_max = 0, 0
cl_max = -100
for i in range(len(data_1['Cl Measured'])):
    if (data_1['Cl Measured'][i] > cl_max):
        cl_max = data_1['Cl Measured'][i]
        stall_min = data_1[xVar_1][i]
        stall_max = data_1[xVar_1][i+1]

ymin = min([min(data_1['Cl Measured']), min(data_1['Cd Measured'])])
ymax = max([max(data_1['Cl Measured']), max(data_1['Cd Measured'])])

plt.vlines(stall_min, ymin, ymax, colors='r', linestyles='solid')
plt.vlines(stall_max, ymin, ymax, colors='r', linestyles='solid')
plt.fill([stall_min, stall_max, stall_max, stall_min], [ymin, ymin, ymax, ymax], color='r', alpha=0.5, label='Begins Stall')

plt.scatter([8], [1.0925343766712312], color = '#842ac9', label='Cl from Integrated Cp') #NEEDS TO BE CHANGED
plt.plot(data_1[xVar_1], data_1['Cl Measured'], color='#0476d4', marker='.', label='Cl Measured')
plt.plot(data_1[xVar_1], data_1['Cd Measured'], color='#26bf96', marker='.', label='Cd Measured')

plt.xlabel('Angle of Attack [deg]')
plt.ylabel('Lift and Drag Coefficient')

plt.grid(b=True, which='both', axis='both')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2, fancybox=True, shadow=True)

plt.savefig('.\IMG\CL_CD_Meas.png', format='png', bbox_inches='tight')
plt.show()
plt.close()

# Integrated Cp and Cd
x_neg = np.array(data_3['x_top'])
x_pos = np.array(data_3['x_bot'][:13])
x_pos = np.array([x_neg[0], *x_pos])
cp_neg = data_3['Cp_top']
cp_pos = data_3['Cp_bot'][:13]
cp_pos = np.array([cp_neg[0], *cp_pos])
xs = np.linspace(0, 1, 256)
dx = xs[1]-xs[0]
cs_neg = np.interp(xs, x_neg, cp_neg)
cs_pos = np.interp(xs, x_pos, cp_pos)

plt.scatter(x_neg, cp_neg, color='#0476d4', marker='.', label='Upper Surface Measured')
plt.scatter(x_pos, cp_pos, color='#26bf96', marker='.', label='Lower Surface Measured')
plt.plot(xs, cs_neg, color='#0476d4', label='Upper Surface Interp.')
plt.plot(xs, cs_pos, color='#26bf96', label='Lower Surface Interp.')

zero = np.zeros(len(xs))
plt.fill_between(xs, cs_neg, cs_pos, color='#a054d6', alpha=0.5)

plt.gca().invert_yaxis()
plt.xlabel('x/c from Front [N.a.]')
plt.ylabel('Pressure Coefficient')

plt.grid(b=True, which='both', axis='both')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2, fancybox=True, shadow=True)

plt.savefig('.\IMG\CD_RE_Meas.png', format='png', bbox_inches='tight')
plt.show()
plt.close()

# Integrate Cl
Cl = 0
for i in range(len(xs)):
    Cl += (cs_pos[i]-cs_neg[i])*dx
print('Integrated Cl: ', Cl)

plt.plot(data_2[xVar_2], data_2['CD Measured'], color='#0476d4', marker='.', label='Cd Measured')

plt.xlabel('Reynolds Number')
plt.ylabel('Drag Coefficient')

plt.grid(b=True, which='both', axis='both')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fancybox=True, shadow=True)

plt.savefig('.\IMG\CD_RE_Meas.png', format='png', bbox_inches='tight')
plt.show()
plt.close()
