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

districts = ['38', '51', '72']
'''Pick out only cluster points and cells'''
for check in districts:
    cluster_points = [idx for idx in all_points if idx[:2].lower() == check.lower()]
    #Find cells that correspond to points
    cluster_cells = []
    for i in range(len(cluster_points)):
        cluster_cells.append(diversions.at[cluster_points[i],'Gridcell'])
    #Get unique values
    cluster_cells = set(cluster_cells)
    #Find corresponding MosartWM cell shortages
    district_shortages_WM_SM = shortages_WM_SM.loc[cluster_cells]
    #Limit StateMod shortages to specific nodes
    district_shortages = shortages.loc[cluster_points]
    #Corvert to m3
    all_shortages = acrefeet_to_m3(district_shortages.values)

    '''Convert to monthly cluster totals, i.e. sum across all cells/points'''
    shortages_monthly = np.sum(all_shortages, axis=0)
    shortages_WM_SM_monthly = seconds_to_month(np.sum(district_shortages_WM_SM.values, axis=0))

    fig, ax = plt.subplots(figsize=(15, 9), dpi=300)

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
    #plt.tight_layout()
    plt.savefig('../figures/fig7_'+check+'.png')
    plt.close()