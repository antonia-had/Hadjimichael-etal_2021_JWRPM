import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils.conversion import *

# Specific gages to generate figures for
gages = [9163500, 9085100, 9050700, 9019500]
#months_available = [298, 282, 249, 272]
# Locations names
names = ['Basin outflow', 'Midstream', 
         'Mid-to-upstream: Below Dillon reservoir',
         'Headwaters: Below Granby lake']

# Read in observations at gages and model simulated
mosartflow = pd.read_csv('../input_data/MosartWM_data/mosartflow_monthly.csv',index_col=0, header=0, usecols = [2,3])
observedflow = pd.read_csv('../input_data/Statemod_data/historicflow.csv',index_col=0, header=None, usecols = [*range(1, 14)])

months_mosart = 29*12
years_record = 105

'''
Generate figure
'''
# Colors for each type of data
observed_clr = '#E09F3E'
mosart_clr='#D62828'
statemod_clr='#003049'

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
for s in range(len(axes.flat)):
    ax = axes.flat[s]
    gage = gages[s]
    #months = months_available[s]
    observedflowvalues = observedflow.loc[gage].values.astype(float)[years_record-5-29:years_record-5,:] #convert to float to deal with nans
    observedflowvalues[observedflowvalues==-999]=np.nan #replace -999 with nans
    observedflowvalues = observedflowvalues[-29:,:]
    observedflowseries = acrefeet_to_m3(observedflowvalues.flatten())*0.000000385802469136

    # observedflow = pd.read_csv('../input_data/External_data/observedflow_'+ str(gage) +'.csv', header=0)
    # observedflowvalues = observedflow['Observed flow (m3/s)']

    statemodflowseries = pd.read_csv('../input_data/StateMod_data/simulatedflow_'+ str(gage) +'.csv', header=0)
    statemodflowvalues = statemodflowseries['Simulated Flow (m3/s)'][-(5+29*12):-5]

    mosartflowseries = mosartflow.loc[gage].values.flatten()
    #mosartflowseries = mosartflowseries[-(13+months):-13]
    mosartflowseries = mosartflowseries[732:1080]

    statemodflowvalues_sorted = np.sort(statemodflowvalues)[::-1]
    #observedflowvalues_sorted = np.sort(observedflowvalues)[::-1]
    observedflowseries_sorted = np.sort(observedflowseries)[::-1]
    mosartflowseries_sorted = np.sort(mosartflowseries)[::-1]
    
    length_models = len(statemodflowvalues)
    #length_gage = len(observedflowvalues)
    P_models = np.arange(100/length_models,100+1/length_models,100/length_models)
    #P_gage = np.arange(100/length_gage,100+1/length_gage,100/length_gage)
    #ax.plot(P_gage, observedflowvalues_sorted, linewidth = 5, c = observed_clr , label = 'Observed flow')
    ax.plot(P_models, observedflowseries_sorted, linewidth = 5, c = observed_clr , label = 'Observed flow')
    ax.plot(P_models, statemodflowvalues_sorted, linewidth = 2, c = statemod_clr, label = 'StateMod flow')
    ax.plot(P_models, mosartflowseries_sorted, linewidth = 2, c = mosart_clr, label = 'MosartWM flow')
    ax.set_yscale('log')
    ax.set_ylabel('Flow ($m^3/s$)', fontsize=16)
    if s>1:
        ax.set_xlabel('Exceedence probability', fontsize=16)
    ax.set_title(names[s], fontsize=18)
    if s==1:
        ax.legend(loc='upper right', fontsize=14)
plt.savefig('../figures/fig5.png')
    
    
