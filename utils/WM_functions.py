from scipy.io import netcdf
import numpy as np
from shapely.geometry import Point

# %%

def read_WM_outputs(file_prix, year_start, year_end):
    # read WM outputs for a range. Outputs in the units of  m3 /day
    demand_ls, supply_ls, Q_ls, Q_riv_ls, year_ls, month_ls = [], [], [], [], [], []

    for year in range(year_start, year_end + 1):
        for month in range(1, 13):
            file = file_prix + str(year) + '-' + format(month, '02d') + '.nc'
            with netcdf.netcdf_file(file, 'r', maskandscale=True) as nc:
                demand_ls.append(nc.variables['WRM_DEMAND0'].data.copy())
                supply_ls.append(nc.variables['WRM_SUPPLY'].data.copy())
                Q_ls.append(nc.variables['QSUR_LIQ'].data.copy() + nc.variables['QSUB_LIQ'].data.copy())
                Q_riv_ls.append(nc.variables['RIVER_DISCHARGE_OVER_LAND_LIQ'].data.copy())
                year_ls.append(year)
                month_ls.append(month)
                lon = nc.variables['lon'].data.copy()
                lat = nc.variables['lat'].data.copy()

    demand = np.concatenate(demand_ls, axis=0)
    supply = np.concatenate(supply_ls, axis=0)
    Q = np.concatenate(Q_ls, axis=0)
    Q_riv = np.concatenate(Q_riv_ls, axis=0)
    deficit = demand - supply
    deficit[deficit < 0] = 0

    # convert to annual as unit m3 /day
    demand = demand * 3600 * 24.0
    supply = supply * 3600 * 24.0
    deficit = deficit * 3600 * 24.0
    Q = Q * 3600 * 24.0
    Q_riv = Q_riv * 3600 * 24.0

    return demand, supply, Q, Q_riv, deficit, year_ls, month_ls

def meshgrid_mask(z, Upper_CO):
    file = '../input_data/MosartWM_data/StateMod_20200715/statemod.mosart.h0.1985-01.nc'
    with netcdf.netcdf_file(file, 'r', maskandscale=True) as nc:
        lon = nc.variables['lon'].data.copy()
        lat = nc.variables['lat'].data.copy()
    mesh_x, mesh_y = np.meshgrid(lon, lat)
    P = np.zeros(np.shape(z))
    #mask out the area outside of your region
    for i in range(mesh_x.shape[0]):
        for j in range(mesh_y.shape[1]):
            if not P[i,j] == np.nan:
                point = Point(mesh_x[i,j], mesh_y[i,j])
                flag = Upper_CO.contains(point).values[0]
                if flag:
                    P[i,j] = 1
    return P

def annual_mean(deficit, mask):
    deficit_co = []
    for i in range(np.shape(deficit)[0]):
        temp = np.squeeze(deficit[i,:,:])
        temp = temp[mask>=1]
        deficit_co.append(np.nansum(temp))

    deficit_co = np.array(deficit_co)
    deficit_co = deficit_co.reshape((12,-1),order='F')
    deficit_annu_co = np.nanmean(deficit_co,axis=0) * 365
    return deficit_annu_co