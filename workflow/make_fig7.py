import pandas as pd
import matplotlib.pyplot as plt
import math
from utils.conversion import *
import matplotlib as mpl

'''Color parameters'''
GCAM_color = 'blue'
USGS_color = 'red'
StateMod_WM_color = 'peru'
StateMod_color = 'black'

'''Shortage data'''
# StateMod
shortages = pd.read_csv('../input_data/StateMod_data/shortages.csv', index_col=0,
                        usecols=[0] + [x for x in range(855+240, 1203)]).astype('float64')
# Mosart StateMod adjusted
shortages_WM_SM = pd.read_pickle('../input_data/MosartWM_data/WM_Outputs_20210309/WMout_SM_shortages_abs.pkl')
shortages_WM_SM = shortages_WM_SM.filter(['0'] + [str(x) for x in range(732+240, 1080)])

'''Diversion locations and corresponding cells'''
diversions = pd.read_csv('../input_data/StateMod_data/modeled_diversions.csv', index_col=0)
all_points = diversions.index.values

'''Pick out only cluster points and cells'''
check = '72'
cluster_points  = [idx for idx in all_points if idx[:2].lower() == check.lower()]
#Find cells that correspond to points
cluster_cells = []
for i in range(len(cluster_points)):
    cluster_cells.append(diversions.at[cluster_points[i],'Gridcell'])
#Get unique values
cluster_cells = set(cluster_cells)
#Find corresponding MosartWM cell shortages
shortages_WM_SM = shortages_WM_SM.loc[cluster_cells]
#Limit StateMod shortages to specific nodes
shortages = shortages.loc[cluster_points]
#Corvert to m3
all_shortages = acrefeet_to_m3(shortages.values)

'''Convert to monthly cluster totals, i.e. sum across all cells/points'''
shortages_monthly = np.sum(all_shortages, axis=0)
shortages_WM_SM_monthly = seconds_to_month(np.sum(shortages_WM_SM.values, axis=0))

'''
Create mesh of shortage magnitude vs time
'''
# Get max shortage across all
maximum_shortage = np.max(all_shortages)
#shortage_brackets = np.logspace(1, math.ceil(np.log10(maximum_shortage)), num=51, endpoint=True)
shortage_brackets = np.linspace(1, maximum_shortage, num=50, endpoint=True)
shortage_density = np.zeros([len(shortage_brackets)-1, len(all_shortages[0, :])])
for i in range(len(all_shortages[0, :])):
    data = all_shortages[:, i]
    counts = np.histogram(data, bins=shortage_brackets)[0]
    shortage_density[:, i] = counts
shortage_density = np.flip(shortage_density, 0)

fig, ax = plt.subplots(figsize=(15, 9), dpi=300)

cmap = mpl.cm.Blues
cmaplist = [cmap(i) for i in range(cmap.N)]
# force the first color entry to be white
cmaplist[0] = (1, 1, 1, 0)
# create the new map
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist, cmap.N)
bounds = [0, 1, 5, np.max(shortage_density)]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
ax2 = ax.twinx()
im = ax2.imshow(shortage_density, cmap=cmap, norm=norm, aspect='auto',
          extent=[1, 108, 0, maximum_shortage/1000000])
im.set_zorder(5)
ax2.set_ylim([0,maximum_shortage/1000000])
ax2.tick_params(axis='y', which='major', labelsize=16)
ax2.set_ylabel('Individual Monthly Deficit (Million $m^3$)', fontsize=18)

ax.plot(range(1,109), shortages_monthly/1000000, color=StateMod_color, zorder=1, linewidth=4, label='StateMod')
ax.plot(range(1,109), shortages_WM_SM_monthly/1000000,color=StateMod_WM_color, zorder=2, linewidth=4, label='StateMod-adjusted MOSART-WM')
ax.set_xticks(range(1,109,12))
ax.set_xticklabels([2000+x for x in range(9)])
ax.tick_params(axis='x',rotation=45)
ax.set_ylabel('Total Cluster Monthly Deficit (Million $m^3$)', fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=16)
ax.set_ylim(bottom=0)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=4)
cbar = fig.colorbar(im, ax=ax2)
cbar.set_label('Occurences of shortage\nlevel among users',size=18)
cbar.ax.tick_params(labelsize=16)
#plt.tight_layout()
plt.savefig('../figures/fig7_'+check+'.svg')
plt.close()