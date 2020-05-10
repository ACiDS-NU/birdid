# -*- coding: utf-8 -*-
"""
Created on Thu May 23 09:50:49 2019

@author: user
"""

# Default imports
import numpy as np
#%matplotlib nbagg
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
#%%
#from mpl_toolkits.basemap import Basemap
np.random.seed(1)
x = 360 * np.random.rand(100)
y = 180 * np.random.rand(100) - 90

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x, y)
#%%
lon = x
lat = y

#%%
provinces_50m = cfeature.NaturalEarthFeature('cultural',
                                             'admin_1_states_provinces_lines',
                                             '50m',
                                             facecolor='none')
land_50m = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cfeature.COLORS['land'])
#%% Create points
def lat_lon_pts(east, west, north, south, grid=1): 
    # Grid is in latitude degrees
    all_lat_pts = []
    all_lon_pts = []
    for ii in np.arange(south, north, grid):
        # Calculate lon points
        pts = round((east - west) * np.cos(ii * np.pi/180) / grid)
        lat_pts = np.linspace(ii,ii,pts)
        lon_pts = np.linspace(west, east, pts)
        for lat, lon in zip(lat_pts, lon_pts):
            all_lat_pts.append(lat)
            all_lon_pts.append(lon)
    return all_lat_pts, all_lon_pts

#%%
USA_48_east = -66
USA_48_west = -125
USA_48_north = 50
USA_48_south = 24
USA_AK_east = -129
USA_AK_west = -190
USA_AK_north = 72
USA_AK_south = 51
USA_HI_east = -154
USA_HI_west = -161
USA_HI_north = 23
USA_HI_south = 18.5
USA_west = -190
USA_east = -66
USA_south = 18.5
USA_north = 72
all_lat_pts = []
all_lon_pts = []
lat_pts, lon_pts = lat_lon_pts(USA_48_east, USA_48_west, USA_48_north, USA_48_south, grid = 1)
all_lat_pts.extend(lat_pts)
all_lon_pts.extend(lon_pts)
lat_pts, lon_pts = lat_lon_pts(USA_AK_east, USA_AK_west, USA_AK_north, USA_AK_south, grid = 1)
all_lat_pts.extend(lat_pts)
all_lon_pts.extend(lon_pts)
lat_pts, lon_pts = lat_lon_pts(USA_HI_east, USA_HI_west, USA_HI_north, USA_HI_south, grid = 0.5)
all_lat_pts.extend(lat_pts)
all_lon_pts.extend(lon_pts)

to_lon = -79.398329  # East is positive.
to_lat = 43.660924

fig = plt.figure(figsize=(18,12))
ax = fig.add_subplot(1, 1, 1,
                     projection=ccrs.PlateCarree(central_longitude=-79))
ax.set_extent([USA_west, USA_east, USA_south, USA_north])
#ax.scatter(lon, lat)
#ax.stock_img()
#ax.add_feature(land_50m)
ax.add_feature(cfeature.LAKES)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.BORDERS)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)
ax.add_feature(provinces_50m, edgecolor='gray')
ax.scatter(all_lon_pts, all_lat_pts, transform=ccrs.PlateCarree(), )
fig.show()


#all_pts = np.array(all_pts).flatten()
#print(all_pts)
