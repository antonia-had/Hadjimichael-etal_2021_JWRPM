#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 16:27:39 2020

@author: Antonia Hadjimichael
"""
import pandas as pd
import cartopy.io.img_tiles as cimgt
import matplotlib.pyplot as plt
import numpy as np
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
import cartopy.crs as ccrs
import matplotlib.patheffects as PathEffects

# StateMod diversion locations
structures = pd.read_csv('./input_data/StateMod_data/modeled_diversions.csv',index_col=0)

# Reservoir locations
WM_reservoirs = pd.read_csv('./input_data/MosartWM_data/dams_storage_capacity.csv', index_col=0)
Statemod_reservoirs = pd.read_csv('./input_data/StateMod_data/reservoirs.csv', index_col=0)

Statemod_total = np.sum(Statemod_reservoirs['cap_mcm'])
Mosart_total = np.sum(WM_reservoirs['cap_mcm'])
missing = Statemod_total-Mosart_total
'''
Map setup
'''
# map extent
extent = [-109.125, -105.625, 38.875, 40.50]

# background map tiles
stamen_terrain = cimgt.Stamen('terrain-background')
# MosartWM gridcells
xlist = np.linspace(-109.125, -105.625, 29)
ylist = np.linspace(38.875, 40.50, 17)
X, Y = np.meshgrid(xlist, ylist)
Xgrid = np.arange(128, 156)
Ygrid = np.arange(112, 128)
Z = np.ones([len(Ygrid), len(Xgrid)])
# Basin shape
shape_feature = ShapelyFeature(Reader('./input_data/Spatial_data/Water_Districts.shp').geometries(),
                               ccrs.PlateCarree(), edgecolor='black', facecolor='None')
# Stream lines
flow_feature = ShapelyFeature(Reader('./input_data/Spatial_data/UCRBstreams.shp').geometries(),
                              ccrs.PlateCarree(), edgecolor='royalblue', facecolor='None')
# USGS gauges
UCRB_gages = pd.read_csv('./input_data/External_data/UCRB_gages.csv', index_col=1)
gages = [9163500, 9085100, 9050700, 9019500]
'''
Figure generation
'''
fig = plt.figure(figsize=(18, 12))
ax = plt.axes(projection=stamen_terrain.crs)
# Set map extent
ax.set_extent(extent, crs=ccrs.PlateCarree())
# Draw background tiles
ax.add_image(stamen_terrain, 9)
# Draw basin
ax.add_feature(shape_feature, facecolor='#a1a384', alpha=0.6)
# Draw streams
ax.add_feature(flow_feature, alpha=0.8, linewidth=1.5, zorder=4)
# Draw grid  
ax.pcolor(X, Y, Z, facecolor='none', edgecolor='grey', linewidth=0.5, transform=ccrs.PlateCarree())
# Draw StateMod reservoirs
SM=ax.scatter(Statemod_reservoirs['lon'], Statemod_reservoirs['lat'],
              marker='.', s=Statemod_reservoirs['cap_mcm']*20,c='#003049', transform=ccrs.PlateCarree(),
              zorder=5, label='StateMod reservoirs')
WM=ax.scatter(WM_reservoirs['lon'], WM_reservoirs['lat'], marker='.',
              s=WM_reservoirs['cap_mcm']*20, c='#D62828', transform=ccrs.PlateCarree(),
              zorder=6, label='MosartWM reservoirs')
for gage in gages:
    ax.scatter(UCRB_gages.loc[gage,'LongDecDeg'], UCRB_gages.loc[gage,'LatDecDeg'], marker='.',
              s=1000, c='white', edgecolors='black', transform=ccrs.PlateCarree(),
              zorder=7)
    ax.text(UCRB_gages.loc[gage,'LongDecDeg'], UCRB_gages.loc[gage,'LatDecDeg'] + 0.05, 'USGS gauge ' + str(gage),
            color='black', size=14, ha='center', va='center', transform=ccrs.PlateCarree(),
            path_effects=[PathEffects.withStroke(linewidth=5, foreground="w", alpha=1)], zorder=10)
# ax.scatter(-109.125, 38.875, marker='.', s=missing*20, c='grey', transform=ccrs.PlateCarree(),
#               zorder=6)
# Draw StateMod nodes
ax.scatter(structures['X'], structures['Y'], marker = '.', s = 200,
           c ='orange', edgecolors='black', transform=ccrs.PlateCarree(), zorder=4)

ax.legend(loc='upper left',fontsize='30')
plt.savefig('./figures/fig1.png')
