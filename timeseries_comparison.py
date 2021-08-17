'''This script generates one three-panel figure,
with four series in each panel:
1) StateMod
2) Out-of-the-box MosartWM
3) USGS-adjusted MosartWM
4) StateMod-adjusted MosartWM.
The the panels are: Total Inflows, Total Demand, Total Storage
Time series is for months October 1979 - September 2009 (i.e. water years 1980-2009)'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils.conversion import *
from utils.WM_functions import *
import geopandas as gpd

'''Color parameters'''
GCAM_color = 'blue'
USGS_color = 'red'
StateMod_WM_color = 'peru'
StateMod_color = 'black'

'''
Annual Flow data
mosartWM flows are all the same
'''
statemodwaterbalace = pd.read_csv('../input_data/Statemod_data/waterbalance.csv',
                                  index_col=0, header=0, usecols = [0,1,6,7,18])
statemod_inflows = statemodwaterbalace['Stream Inflow'].values[71:101]

# read USGS-adjusted and statemod-adjusted WM data
#GCAM dataset is missing
#demand, supply, Q, Q_riv, deficit, year_ls, month_ls = read_WM_outputs('../input_data/MosartWM_data/OutBoxGCAM/', 1980, 2008)
demand_usgs, supply_usgs, Q_usgs, Q_riv_usgs, deficit_usgs, year_ls, month_ls = read_WM_outputs('../input_data/MosartWM_data/USGS_20200623/usgs.mosart.h0.', 1981, 2009)
demand_statemod, supply_statemod, Q_statemod, Q_riv_statemod, deficit_statemod, year_ls, month_ls = read_WM_outputs('../input_data/MosartWM_data/StateMod_20200715/statemod.mosart.h0.', 1980, 2008)

'''
Figure out mask of netCDF MosartWM data
'''
supply_hist = np.nanmean(supply_usgs, axis=0)
water_districts_dissolve = gpd.read_file('../input_data/Spatial_data/Water_Districts_disolve.shp')
mask = meshgrid_mask(supply_hist, water_districts_dissolve)

Q_annu_co_usgs = annual_mean(Q_usgs, mask)
Q_annu_co_statemod = annual_mean(Q_statemod, mask)

'''Supply data'''
# StateMod (water years 1980-2009)
statemod_demands = pd.read_csv('../input_data/StateMod_data/demands_types.csv',
                               index_col=0, usecols=[0, 1] + [x for x in range(856, 1216)])
statemod_demands_toconsume = statemod_demands.copy()
ratios = {}
ratios['Irrigation'] = 1023.0 / 2464
ratios['M&I'] = 65.0 / 466
ratios['Reservoir'] = 0
ratios['Transbasin'] = 1
ratios['Other'] = 0
for j, row in statemod_demands_toconsume.iterrows():
    ratio = ratios[row['Type']]
    values = row[statemod_demands_toconsume.columns!='Type'].values
    statemod_demands_toconsume.loc[j, statemod_demands_toconsume.columns!='Type'] =values*ratio
statemod_demands.drop(['Type'],axis=1, inplace=True)
statemod_demands_toconsume.drop(['Type'],axis=1, inplace=True)
statemod_shortages = pd.read_csv('../input_data/StateMod_data/shortages.csv',
                               index_col=0, usecols=[0] + [x for x in range(855, 1215)]).astype('float64')
statemod_CU_annual = statemodwaterbalace['CU (1)'].values[71:101]
# Mosart out-of-the-box (years 1980-2008)
demands_WM = pd.read_pickle('../input_data/MosartWM_data/OutBoxGCAM/demands.pkl')
# Mosart USGS adjusted (years 1981-2009)
demands_WM_USGS = pd.read_pickle('../input_data/MosartWM_data/WM_Outputs_20210309/WMout_USGS_demands.pkl')
# Mosart StateMod adjusted (water years 1980-2009)
demands_WM_SM = pd.read_pickle('../input_data/MosartWM_data/WM_Outputs_20210309/WMout_SM_demands.pkl')
demands_WM_SM = demands_WM_SM.filter(['0'] + [str(x) for x in range(730, 1090)])

'''
Convert to monthly
Sum across all cells/points and perform necessary conversions'''
demands_WM_monthly = seconds_to_month(np.sum(demands_WM.values, axis=0))
demands_WM_USGS_monthly = seconds_to_month(np.sum(demands_WM_USGS.values, axis=0))
demands_WM_SM_monthly = seconds_to_month(np.sum(demands_WM_SM.values, axis=0))

statemod_demands_monthly = acrefeet_to_m3(np.sum(statemod_demands.values, axis=0))
statemod_demandstoconsume_monthly = acrefeet_to_m3(np.sum(statemod_demands_toconsume.values, axis=0))
statemod_shortages_monthly = acrefeet_to_m3(np.sum(statemod_shortages.values, axis=0))

'''Convert to annual'''
demands_WM_annual = month_to_annual(demands_WM_monthly)
demands_WM_USGS_annual = month_to_annual(demands_WM_USGS_monthly)
demands_WM_SM_annual = month_to_annual(demands_WM_SM_monthly)
statemod_demands_annual = month_to_annual(statemod_demands_monthly)
statemod_demands_annualtoconsume = month_to_annual(statemod_demandstoconsume_monthly)
statemod_shortages_annual = month_to_annual(statemod_shortages_monthly)
statemod_supply_annual = statemod_demands_annual - statemod_shortages_annual

'''Reservoir storage data'''
statemod_storage = pd.read_csv('../input_data/Statemod_data/storage_eom_DEC.csv',index_col=0, header=None)
mosartWM_storage = pd.read_pickle('../input_data/MosartWM_data/WM_Outputs_20210309/WMout_SM_storage.pkl')

statemod_storage_total = acrefeet_to_m3(np.sum(statemod_storage.values, axis=1))[71:101]
mosartWM_storage_dec = mosartWM_storage[mosartWM_storage.columns[11::12]]
mosartWM_storage_total = np.sum(mosartWM_storage_dec.values, axis=0)[61:91]

'''Generate timeseries figure'''

fig, axes = plt.subplots(3, 1, figsize=(12,9))

ax = axes[0]
ax.plot(range(1980, 2010), statemod_inflows/1000000, color=StateMod_color,label='StateMod')
ax.plot(range(1981, 2010), Q_annu_co_usgs/1000000,color=USGS_color, label='USGS-adjusted MOSART-WM')
ax.plot(range(1980, 2009), Q_annu_co_statemod/1000000,color=StateMod_WM_color, label='StateMod-adjusted MOSART-WM')
ax.set_xticks(range(1980, 2010, 2))
ax.set_xticklabels([1980+x for x in range(0, 30, 2)])
ax.tick_params(axis='x', rotation=45, labelsize=8)
ax.set_ylabel('Total infow\n($10^6$ $m^3$)')

ax = axes[1]
ax.plot(range(1980, 2010), statemod_demands_annual/1000000, color=StateMod_color, label='StateMod demands')
ax.plot(range(1980, 2010), statemod_demands_annualtoconsume/1000000, color=StateMod_color, linestyle='dashed', label='StateMod consumptive demands')
ax.plot(range(1980, 2009), demands_WM_annual/1000000, color=GCAM_color, label='GCAM-informed MOSART-WM consumptive demands')
ax.plot(range(1980, 2010), demands_WM_USGS_annual/1000000,color=USGS_color, label='USGS-adjusted MOSART-WM consumptive demands')
ax.plot(range(1980, 2010), demands_WM_SM_annual/1000000,color=StateMod_WM_color, label='StateMod-adjusted MOSART-WM consumptive demands')
ax.set_xticks(range(1980, 2010, 2))
ax.set_xticklabels([1980+x for x in range(0, 30,2)])
ax.tick_params(axis='x',rotation=45, labelsize=8)
ax.set_ylabel('Annual Demand\n($10^6$ $m^3$)')
handles, labels = ax.get_legend_handles_labels()
ax.grid()

ax = axes[2]
ax.plot(range(1980, 2010), statemod_storage_total/1000000, color=StateMod_color,label='StateMod')
ax.plot(range(1980, 2010), mosartWM_storage_total/1000000, color=StateMod_WM_color, label='StateMod-adjusted MOSART-WM')
ax.set_xticks(range(1980, 2010, 2))
ax.set_xticklabels([1980+x for x in range(0, 30, 2)])
ax.tick_params(axis='x',rotation=45, labelsize=8)
ax.set_ylabel('End-of-year\ntotal storage\n($10^6$ $m^3$)')

fig.legend(handles, labels,
           ncol=1, fontsize='small',
           loc='center right')
plt.subplots_adjust(right=0.85)
plt.savefig('../figures/timeseries_comparison_annual.svg')