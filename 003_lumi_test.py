import pandas as pd
import sys
sys.path.append('/afs/cern.ch/eng/tracking-tools/modules')
from pymask import luminosity as lumi
ip8_lumi=pd.read_parquet('./data/lumi_dict_ip8.parquet')
ip8_dict=ip8_lumi.iloc[0].to_dict()
print(ip8_dict)

print(f'L is :{lumi.L(**ip8_dict)}')

ip8_dict_new=ip8_dict.copy()


for ii in ['x_','py_','dx_', 'dy_','dpx_','dpy_','alpha_x','alpha_y', 'deltap_p0_','px_']:
	ip8_dict_new[ii+'1']=0
	ip8_dict_new[ii+'2']=0
	print(f'Zeroing also {ii}, L is :{lumi.L(**ip8_dict_new)}')

ip8_dict_new['y_1']=-0.02939387537*1e-3;
ip8_dict_new['y_2']=+0.02939387537*1e-3;
print(f'With Sofia y-positions, L is :{lumi.L(**ip8_dict_new)}')
sofia_df=pd.read_pickle('./data/input_params_2484_2.5.pickle')
ip8_dict_new['N1']=sofia_df.iloc[0].Intensity 
ip8_dict_new['N2']=sofia_df.iloc[0].Intensity 
print(f'With Sofia bunch population, L is :{lumi.L(**ip8_dict_new)}')

# Using the closed form

import numpy as np

def myLumi(f,nb, N1, N2, energy_tot1,epsilon_x1, beta_x1, y_1):
	gamma=energy_tot1/.938
	sigma=np.sqrt(epsilon_x1/gamma*beta_x1)
	return f*N1*N2*nb/4/np.pi/sigma**2/1e4*np.exp(-(2*y_1)**2/(2*np.sqrt(2*sigma**2)**2))

print(myLumi(ip8_dict_new['f'],
ip8_dict_new['nb'],
ip8_dict_new['N1'],
ip8_dict_new['N2'],
ip8_dict_new['energy_tot1'],
ip8_dict_new['epsilon_x1'],
ip8_dict_new['beta_x1'],
ip8_dict_new['y_1'],
))


print(myLumi(ip8_dict_new['f'],
ip8_dict_new['nb'],
ip8_dict_new['N1'],
ip8_dict_new['N2'],
ip8_dict_new['energy_tot1'],
ip8_dict_new['epsilon_x1'],
ip8_dict_new['beta_x1'],
0.03304121488976953*1e-3,
))

