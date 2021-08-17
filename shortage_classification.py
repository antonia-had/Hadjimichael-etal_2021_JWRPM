import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import itertools
import scipy.stats
import math
from mpl_toolkits.mplot3d import Axes3D

model='mosart_statemod' #'statemod' or 'mosart' or 'mosart_usgs' or 'mosart_statemod'

# Function to round up to nearest 10
roundup = lambda a : int(math.ceil(a / 10.0)) * 10

# Function to count instances in order
def shortage_duration(sequence, threshold):
    cnt_shrt = [sequence[i]>threshold for i in range(len(sequence))] # Returns a list of True values when there's a shortage
    shrt_dur = [ sum( 1 for _ in group ) for key, group in itertools.groupby( cnt_shrt ) if key ] # Counts groups of True values
    return shrt_dur

# Fuction to find max number of independent consecutive shortages from list
def maxlength(listoflists):
    mx=0
    for lst in listoflists:
        for x in lst:
            if len(x)>mx:
                mx=len(x)
    return mx

# Read in monthly demands and shortages and calculate ratios
if model == 'statemod':
    shortages = pd.read_csv('../input_data/StateMod_data/shortages.csv', index_col=0, usecols=[0] + [x for x in range(855, 1203)]).astype('float64')
    demands = pd.read_csv('../input_data/StateMod_data/demands.csv', index_col=0, usecols=[0] + [x for x in range(855, 1203)]).astype('float64')
elif model == 'mosart':
    shortages = pd.read_pickle('../input_data/MosartWM_data/OutBoxGCAM/shortages.pkl')
    demands = pd.read_pickle('../input_data/MosartWM_data/OutBoxGCAM/demands.pkl')
elif model == 'mosart_usgs':
    shortages = pd.read_pickle('../input_data/MosartWM_data/WM_Outputs_20210309/WMout_USGS_shortages_abs.pkl')
    demands = pd.read_pickle('../input_data/MosartWM_data/WM_Outputs_20210309/WMout_USGS_demands.pkl')
elif model == 'mosart_statemod':
    shortages = pd.read_pickle('../input_data/MosartWM_data/WM_Outputs_20210309/WMout_SM_shortages_abs.pkl')
    demands = pd.read_pickle('../input_data/MosartWM_data/WM_Outputs_20210309/WMout_SM_demands.pkl')
    shortages = shortages.filter(['0'] + [str(x) for x in range(730, 1090)])
    demands = demands.filter(['0'] + [str(x) for x in range(730, 1090)])

shortagesmonthly = shortages.values
demandsmonthly = demands.values

n = 12
if model=='statemod':
    months = len(shortages.loc['3600507'])
else:
    months = len(shortages.loc['x148y117'])

structures = len(shortages)

# Convert to annual demands and shortages and calculate ratios
shortagesannual = np.zeros([structures, int(len(shortagesmonthly[0, :])/n)])
demandsannual = np.zeros([structures, int(len(shortagesmonthly[0, :])/n)])
for i in range(structures):
    shortagesannual[i, :] = np.add.reduceat(shortagesmonthly[i, :], np.arange(0, len(shortagesmonthly[i, :]), n))
    demandsannual[i, :] = np.add.reduceat(demandsmonthly[i, :], np.arange(0, len(demandsmonthly[i, :]), n))

ratiosannual = np.divide(shortagesannual, demandsannual, out=np.zeros_like(shortagesannual), where=demandsannual!=0)*100

#Bin to shortages of 0-10, 10-20, ..., 90-100%
#annualbins = np.vectorize(roundup)(ratiosannual)

# Calculate frequencies and durations per magnitude
magnitudes = np.arange(10, 110, 10)
shortagedurations = [[] for s in range(structures)] # create list to store durations for every structure
frequencies = np.zeros([structures, len(magnitudes)]) # create array to store frequency per magnitude for each structure
for i in range(structures):
    for m in range(len(magnitudes)):
        # calculate frequency of occurrence for each magnitude
        frequencies[i, m] = 100-scipy.stats.percentileofscore(ratiosannual[i, :], magnitudes[m], kind='strict')
        # calculate duration for each magnitude
        shrt_dur = shortage_duration(ratiosannual[i, :], magnitudes[m])
        # only add to list if at least one occurrence
        if shrt_dur:
            shortagedurations[i].append(shrt_dur)
        else:
            shortagedurations[i].append([0])

'''
Because the arrays of frequences and durations are not the same sizes,
we need to create a dataframe to store all data to plot
'''
IDs = list(shortages.index)
tuples = list(zip(np.repeat(IDs, 10), np.tile(magnitudes,structures)))
index = pd.MultiIndex.from_tuples(tuples, names=['structure', 'percentile'])
data_to_plot = pd.DataFrame(data=np.tile(magnitudes, structures), columns=['Mags'], index=index)
data_to_plot['Freqs'] = frequencies.flatten()
#Create columns of zero durations for up to the largest number of separate durations for any structure
for j in range(maxlength(shortagedurations)):
    data_to_plot[j] = 0
for i in range(structures):
    for m in range(len(magnitudes)):
        locationdurations = shortagedurations[i][m]
        for d in range(len(locationdurations)):
            data_to_plot.loc[(IDs[i], magnitudes[m]), d] = locationdurations[d]

'''Heatmap calculations
'''
# Count classified users per magnitude and frequency combination
freq_levels = np.arange(100, 0, -5)
frequencycounts = np.zeros([len(freq_levels), len(magnitudes)])
for i in range(len(freq_levels)):
    for j in range(len(magnitudes)):
        frequencycounts[i,j] = len(np.where(frequencies[:, j] >= freq_levels[i])[0])
# Count classified users per magnitude and duration combination
durationcounts = np.zeros([30, len(magnitudes)])
for j in range(len(magnitudes)):
    data = data_to_plot.loc[data_to_plot['Mags'] == magnitudes[j]].drop(['Mags', 'Freqs'], axis=1)
    for i in range(30):
        durationcounts[i, j] = len(data[(data.values >= i+1).any(1)])

'''Plot 3D bars with classification percentages
'''
fig=plt.figure(figsize=(14.5,8))
#Magnitude vs frequency plot
ax1 = fig.add_subplot(121, projection='3d')
X, Y = np.meshgrid(magnitudes, freq_levels)
xpos = X.flatten()
ypos = Y.flatten()
zpos = np.zeros(len(xpos))    
dx = np.ones(len(xpos))
dy = np.ones(len(ypos))*3.33
# Plot bar lengths as counts of magnitude and frequency
dzfreq = frequencycounts.flatten()*100/len(IDs)
ax1.bar3d(xpos, ypos, zpos, dx, dy, dzfreq, color='dodgerblue')
ax1.set_xlabel('Shortage magnitude (%)', fontsize=12)
ax1.set_ylabel('Frequency of occurence (%)', fontsize=12)
ax1.set_zlabel('Users classified (%)', fontsize=12)
ax1.view_init(elev=23, azim=-27)
#Magnitude vs duration plot
ax2 = fig.add_subplot(122, projection='3d')
# Plot bar lengths as counts of magnitude and duration
X, Y = np.meshgrid(magnitudes, np.arange(1,31))
xpos = X.flatten()
ypos = Y.flatten()
zpos = np.zeros(len(xpos))
dx = np.ones(len(xpos))
dy = np.ones(len(ypos))*0.8
dzdur = durationcounts.flatten()*100/len(IDs)
ax2.bar3d(xpos, ypos, zpos, dx, dy, dzdur, color='dodgerblue')
ax1.set_zlim3d(top=100)
ax2.set_zlim3d(top=100)
ax2.set_xlabel('Shortage magnitude (%)', fontsize=12)
ax2.set_ylabel('Years of continuous occurence (#)', fontsize=12)
ax2.set_zlabel('Users classified (%)', fontsize=12)
ax2.view_init(elev=23, azim=-27)
plt.savefig('../figures/shortage_classification_'+model+'_per.png')

'''
# Calculate metrics for entire basin
'''
basinshortages = np.sum(shortagesannual, axis=0)
basindemands = np.sum(demandsannual, axis=0)
basinratios = basinshortages*100/basindemands
basinfrequencies = [100-scipy.stats.percentileofscore(basinratios, mag, kind='strict') for mag in magnitudes]
basin_durations = [shortage_duration(basinratios, mag) for mag in magnitudes]

# # Create dataframe to store all data to plot
# data_to_plot = pd.DataFrame(data=magnitudes, columns=['Mags'])
# data_to_plot['Freqs'] = np.zeros(len(magnitudes))
# for j in range(maxlength(basin_durations)):
#     data_to_plot[j] = 0
# for m in range(len(magnitudes)):
#     locationdurations = basin_durations[m]
#     for d in range(len(locationdurations)):
#         data_to_plot.loc[m, d] = locationdurations[d]
#
# '''Heatmap calculations
# '''
# # Count classified users per magnitude and frequency combination
# frequencycounts = np.zeros([len(freq_levels), len(magnitudes)])
# for i in range(len(freq_levels)):
#     for j in range(len(magnitudes)):
#         frequencycounts[i,j] = len(np.where(basinfrequencies[j] >= freq_levels[i])[0])
# # Count classified users per magnitude and duration combination
# durationcounts = np.zeros([30, len(magnitudes)])
# for j in range(len(magnitudes)):
#     data = data_to_plot.loc[data_to_plot['Mags'] == magnitudes[j]].drop(['Mags', 'Freqs'], axis=1)
#     for i in range(30):
#         durationcounts[i, j] = len(data[(data.values >= i+1).any(1)])

print('Using ' + model + ' the basin had the following shortage frequencies:')
print(basinfrequencies)
print('at these durations')
print(basin_durations)