# -*- coding: utf-8 -*-
"""
Created on Mon May 27 20:57:20 2019

@author: user
"""

import pandas as pd
#import dask.dataframe as dd
import numpy as np
import time
import datetime
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
#%%
def getRoundedThresholdv1(a, MinClip):
    if a > 100:
        a = a - 360
    return round(a / MinClip) * MinClip

def getWeekNumber(date):
    return int(date.strftime("%V"))

def filterChunk(chunk, checklists, group_identifier):
    chunk = chunk[chunk['COUNTRY CODE']=='US']
    chunk = chunk[chunk['PROTOCOL TYPE']!='Incidental']
    chunk = chunk[(chunk['OBSERVATION DATE'] > '2015-01-01') & (chunk['OBSERVATION DATE'] < '2018-12-31')]
#    chunk_nan = chunk[chunk['GROUP IDENTIFIER'].isnull()]
#    chunk_dup = chunk[~chunk['GROUP IDENTIFIER'].isnull()]
    if len(chunk) > 0:
        print(len(chunk))

    for index, COUNTRY_CODE, STATE, COUNTY, LATITUDE, LONGITUDE, OBSERVATION_DATE, OBSERVER_ID, SAMPLING_EVENT_IDENTIFIER, _, GROUP_IDENTIFIER in chunk.itertuples():
#        if GROUP_IDENTIFIER not in group_identifier:
##            if OBSERVER_ID != group_identifier[GROUP_IDENTIFIER]:
##                continue
##        else:
#            group_identifier[GROUP_IDENTIFIER] = OBSERVER_ID
            
        if SAMPLING_EVENT_IDENTIFIER in checklists:
            if checklists[SAMPLING_EVENT_IDENTIFIER]['OBSERVER ID'] == OBSERVER_ID:
                checklists[SAMPLING_EVENT_IDENTIFIER]['SPECIES COUNT'] += 1
        else:
            checklists[SAMPLING_EVENT_IDENTIFIER] = {'OBSERVER ID': OBSERVER_ID,
                      'LATITUDE': LATITUDE, 'LONGITUDE': LONGITUDE, 
                      'OBSERVATION_DATE': OBSERVATION_DATE, 'STATE': STATE,
                      'COUNTY': COUNTY, 'GROUP_IDENTIFIER': GROUP_IDENTIFIER,
                      'SPECIES COUNT': 1}
            
#    for index, COUNTRY_CODE, STATE, COUNTY, LATITUDE, LONGITUDE, OBSERVATION_DATE, OBSERVER_ID, SAMPLING_EVENT_IDENTIFIER, _, GROUP_IDENTIFIER in chunk_nan.itertuples():
#        if SAMPLING_EVENT_IDENTIFIER in checklists:
#            if checklists[SAMPLING_EVENT_IDENTIFIER]['OBSERVER ID'] == OBSERVER_ID:
#                checklists[SAMPLING_EVENT_IDENTIFIER]['SPECIES COUNT'] += 1
#        else:
#            checklists[SAMPLING_EVENT_IDENTIFIER] = {'OBSERVER ID': OBSERVER_ID,
#                      'LATITUDE': LATITUDE, 'LONGITUDE': LONGITUDE, 
#                      'OBSERVATION_DATE': OBSERVATION_DATE, 'STATE': STATE,
#                      'COUNTY': COUNTY, 'SPECIES COUNT': 1}
#            checklists[row['SAMPLING EVENT IDENTIFIER']] = {}
#            checklists[row['SAMPLING EVENT IDENTIFIER']]['OBSERVER ID'] = row['OBSERVER ID']
#            checklists[row['SAMPLING EVENT IDENTIFIER']]['LATITUDE'] = row['LATITUDE']
#            checklists[row['SAMPLING EVENT IDENTIFIER']]['LONGITUDE'] = row['LONGITUDE']
#            checklists[row['SAMPLING EVENT IDENTIFIER']]['STATE'] = row['STATE']
#            checklists[row['SAMPLING EVENT IDENTIFIER']]['COUNTY'] = row['COUNTY']
#            checklists[row['SAMPLING EVENT IDENTIFIER']]['SPECIES COUNT'] = 1

#    for index, row in chunk[chunk['GROUP IDENTIFIER'].isnull()].iterrows():
#        if row['SAMPLING EVENT IDENTIFIER'] in checklists:
#            if checklists[row['SAMPLING EVENT IDENTIFIER']]['OBSERVER ID'] == row['OBSERVER ID']:
#                checklists[row['SAMPLING EVENT IDENTIFIER']]['SPECIES COUNT'] += 1
#        else:
#            checklists[row['SAMPLING EVENT IDENTIFIER']] = {'OBSERVER ID': row['OBSERVER ID'],
#                      'LATITUDE': row['LATITUDE'], 'LONGITUDE': row['LONGITUDE'], 'STATE': row['STATE'],
#                      'COUNTY': row['COUNTY'], 'SPECIES COUNT': 1}
#                       
#    for index, row in chunk[~chunk['GROUP IDENTIFIER'].isnull()].iterrows():
#        if row['GROUP IDENTIFIER'] in group_identifier:
#            if row['OBSERVER ID'] != group_identifier[row['GROUP IDENTIFIER']]:
#                continue
#        else:
#            group_identifier[row['GROUP IDENTIFIER']] = row['OBSERVER ID']
#            
#        if row['SAMPLING EVENT IDENTIFIER'] in checklists:
#            if checklists[row['SAMPLING EVENT IDENTIFIER']]['OBSERVER ID'] == row['OBSERVER ID']:
#                checklists[row['SAMPLING EVENT IDENTIFIER']]['SPECIES COUNT'] += 1
#        else:
#            checklists[row['SAMPLING EVENT IDENTIFIER']] = {'OBSERVER ID': row['OBSERVER ID'],
#                      'LATITUDE': row['LATITUDE'], 'LONGITUDE': row['LONGITUDE'], 'STATE': row['STATE'],
#                      'COUNTY': row['COUNTY'], 'SPECIES COUNT': 1}
#            checklists[row['SAMPLING EVENT IDENTIFIER']] = {}
#            checklists[row['SAMPLING EVENT IDENTIFIER']]['OBSERVER ID'] = row['OBSERVER ID']
#            checklists[row['SAMPLING EVENT IDENTIFIER']]['LATITUDE'] = row['LATITUDE']
#            checklists[row['SAMPLING EVENT IDENTIFIER']]['LONGITUDE'] = row['LONGITUDE']
#            checklists[row['SAMPLING EVENT IDENTIFIER']]['STATE'] = row['STATE']
#            checklists[row['SAMPLING EVENT IDENTIFIER']]['COUNTY'] = row['COUNTY']
#            checklists[row['SAMPLING EVENT IDENTIFIER']]['SPECIES COUNT'] = 1
    return checklists, group_identifier
            

#%%
chunk_size = 3e6
#df = pd.read_csv("test_ebird.txt", delimiter='\t')
#df_reader = pd.read_csv("E:\ebd_sampling_relApr-2019\ebd_relApr-2019.txt.gz", delimiter='\t', iterator=True, compression='gzip', chunksize=3e6, usecols=['COUNTRY CODE', 'STATE', 'COUNTY', 'LATITUDE', 'LONGITUDE', 'OBSERVATION DATE', 'OBSERVER ID', 'PROTOCOL TYPE', 'SAMPLING EVENT IDENTIFIER', 'GROUP IDENTIFIER'])
df_reader = pd.read_csv("E:\ebd_sampling_relApr-2019\ebd_relApr-2019.txt", delimiter='\t', iterator=True, chunksize=chunk_size, usecols=['COUNTRY CODE', 'STATE', 'COUNTY', 'LATITUDE', 'LONGITUDE', 'OBSERVATION DATE', 'OBSERVER ID', 'PROTOCOL TYPE', 'SAMPLING EVENT IDENTIFIER', 'GROUP IDENTIFIER'], engine='c')
#df_reader = pd.read_csv("test_ebird.txt", delimiter='\t', iterator=True, chunksize=300000, usecols=['COUNTRY CODE', 'STATE', 'COUNTY', 'LATITUDE', 'LONGITUDE', 'OBSERVATION DATE', 'OBSERVER ID', 'PROTOCOL TYPE', 'SAMPLING EVENT IDENTIFIER', 'GROUP IDENTIFIER'])
#%%
checklists={}
group_identifier = {}
counter = 1
start = time.time()
for chunk in df_reader:
    end = time.time()
    print("chunk time: {:.3f}".format(end - start))
    start = time.time()
    print(counter * chunk_size)
    checklists, group_identifier = filterChunk(chunk, checklists, group_identifier)
    counter += 1
    end = time.time()
    print("filter time: {:.3f}".format(end - start))
    print(len(checklists))
    start = time.time()
    
#%%
f = open("Checklists_dup.pkl","wb")
pickle.dump(checklists,f)
f.close()
#%%
f = open("Checklists_dup.pkl","rb")
checklists = pickle.load(f)
f.close()
#%%
text_file = open("E:\ebd_sampling_relApr-2019\ebird_species_per_checklist_dup.txt", "w")
text_file.write('CHECKLIST\tOBSERVATION_DATE\tOBSERVER_ID\tLATITUDE\tLONGITUDE\tSTATE\tCOUNTY\tGROUP_IDENTIFIER\tSPECIES_COUNT\n')
counter = 0
for key, details in checklists.items():
    counter += 1
    text_file.write('{:s}\t{:s}\t{:s}\t{:s}\t{:s}\t{:s}\t{:s}\t{:s}\t{:s}\n'.format(key, 
                    str(details['OBSERVATION_DATE']), str(details['OBSERVER ID']), 
                    str(details['LATITUDE']), str(details['LONGITUDE']), 
                    str(details['STATE']), str(details['COUNTY']), 
                    str(details['GROUP_IDENTIFIER']), str(details['SPECIES COUNT'])))
    if (counter % 100000 == 0):
        print(counter)
text_file.close()

#%%
#df = pd.read_csv("E:\ebd_sampling_relApr-2019\ebird_species_per_checklist_dup.txt", delimiter='\t', usecols=['OBSERVATION_DATE'], dtype=str)
#df = pd.read_csv("E:\ebd_sampling_relApr-2019\ebird_species_per_checklist_dup.txt", delimiter='\t', usecols=['OBSERVATION_DATE','LATITUDE','LONGITUDE','SPECIES_COUNT'], dtype=str)
#                 dtype=[('OBSERVATION_DATE',str),('LATITUDE',float), ('LONGITUDE',float),('SPECIES_COUNT',int)], engine='c')
#df = pd.read_csv("ebird_species_per_checklist.txt", delimiter='\t', usecols=['LATITUDE','LONGITUDE','SPECIES_COUNT'], dtype=float, engine='c')
#df = pd.read_csv("ebird_species_per_checklist.txt", delimiter='\t', usecols=['OBSERVATION_DATE','STATE','COUNTY','SPECIES_COUNT'], dtype=str, engine='c')
df = pd.read_csv("E:\ebd_sampling_relApr-2019\ebird_species_per_checklist_dup.txt", delimiter='\t', usecols=['OBSERVATION_DATE','STATE','COUNTY','SPECIES_COUNT'], dtype=str, engine='c')
#%%
df['OBSERVATION_DATE']=pd.to_datetime(df['OBSERVATION_DATE'])
#df['LONGITUDE']=pd.to_numeric(df['LONGITUDE'])
#df['LATITUDE']=pd.to_numeric(df['LATITUDE'])
df['SPECIES_COUNT']=pd.to_numeric(df['SPECIES_COUNT'])

#%%
sc = df['SPECIES_COUNT'].value_counts().to_dict()
#%%
sc_arr = np.array([(key, val) for key, val in sc.items()])
#%%
print(sc_arr)
fig = plt.figure(figsize=(18,12))
#ax = fig.add_subplot(1,1,1)
plt.scatter(sc_arr[:,0],sc_arr[:,1])
#for tick in plt.xaxis.get_major_ticks():
#    tick.label.set_fontsize(25) 
#for tick in plt.yaxis.get_major_ticks():
#    tick.label.set_fontsize(25) 
plt.xticks(fontsize=26)
plt.yticks(fontsize=26)
plt.xlabel('Species on a checklist', fontsize=26)
plt.ylabel('Number of checklists', fontsize=26)
plt.show()
#%%
df_1000 = df.iloc[0:1000]
#%%
start = time.time()
lat_round = []
lon_round = []
week_num = []
for index, OBS_DATE, LATITUDE, LONGITUDE, SPECIES_COUNT in df.itertuples():
    if index % 100000 == 0:
        print(index)
    lat_round.append(getRoundedThresholdv1(LATITUDE, 0.5))
    lon_round.append(getRoundedThresholdv1(LONGITUDE, 0.5))
    week_num.append(getWeekNumber(OBS_DATE))
#    df_1000['lat_round'] = df_1000.apply(lambda row: getRoundedThresholdv1(row['LATITUDE'], 0.02), axis=1)
#    df_1000['lon_round'] = df_1000.apply(lambda row: getRoundedThresholdv1(row['LONGITUDE'], 0.02), axis=1)
df_round = pd.DataFrame({'lat_round': lat_round, 'lon_round': lon_round, 'lat_round2': lat_round, 'lon_round2': lon_round, 'week_num': week_num, 'week_num2': week_num})
df_r = pd.concat([df, df_round], axis=1)
end = time.time()
print("Elapsed time: {:.3f} s".format(end - start))
#%%
lat_lon = df_r.groupby(['lat_round', 'lon_round','week_num']).mean()
#%%
week_num = []
for index, OBS_DATE, STATE, COUNTY, SPECIES_COUNT in df.itertuples():
    if index % 100000 == 0:
        print(index)
    week_num.append(getWeekNumber(OBS_DATE))
df_round = pd.DataFrame({'week_num': week_num})
df_r = pd.concat([df, df_round], axis=1)
state_county = df_r.groupby(['STATE','COUNTY','week_num']).mean()
#%%
provinces_10m = cfeature.NaturalEarthFeature('cultural',
                                             'admin_1_states_provinces_lines',
                                             '10m', facecolor='none')
#land_10m = cfeature.NaturalEarthFeature('physical', 'land', '10m',
#                                        edgecolor='k',
#                                        facecolor=cfeature.COLORS['land'])
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
USA_south = 15
USA_north = 72

to_lon = -79.398329  # East is positive.
to_lat = 43.660924

fig = plt.figure(figsize=(22,12))
ax = fig.add_subplot(1, 1, 1,
                     projection=ccrs.PlateCarree(central_longitude=-79))
ax.set_extent([USA_west, USA_east, USA_south, USA_north])
ax.set_extent([USA_48_west-1, USA_48_east+1, USA_48_south-1, USA_48_north+1])
#ax.scatter(lon, lat)
#ax.stock_img()
#ax.add_feature(provinces_10m)
ax.add_feature(cfeature.LAKES, alpha=0.5)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_wms(wms='http://vmap0.tiles.osgeo.org/wms/vmap0',
           layers=['basic'])

#week_this = int(datetime.datetime.today().strftime("%V"))
week_this = 30
plt.scatter(lat_lon[lat_lon['week_num2']==week_this]['lon_round2'].values+79, 
            lat_lon[lat_lon['week_num2']==week_this]['lat_round2'].values, 
            c = lat_lon[lat_lon['week_num2']==week_this]['SPECIES_COUNT'].values, 
            cmap=cm.winter, vmin=1, vmax=25, 
            marker='s', s = 90, alpha=0.7)
ax.add_feature(provinces_10m, edgecolor='777', alpha = 0.7)
ax.add_feature(cfeature.BORDERS, linewidth = 1, edgecolor = '777', alpha = 0.6)
ax.add_feature(cfeature.RIVERS, linewidth = 3, edgecolor ='r', alpha = 0.6)

cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=26) 
plt.xticks(fontsize=26)
plt.yticks(fontsize=26)
plt.show()
#%%
lat_lon.to_csv('Species_per_checklist_by_week.csv')
#%%
state_county.to_csv('Species_per_checklist_by_week_state_county_dup.csv')
#%%
lat_lon = pd.read_csv('Species_per_checklist_by_week.csv')
#%%

#%%
week_this = int(datetime.datetime.today().strftime("%V"))
#lat = 40.0
lat_this, lon_this = 41.9631676, -87.6333702147878

lat_lon[lat_lon['week_num2'] == week_this][lat_lon['lat_round2']==getRoundedThresholdv1(lat_this, 0.5)][lat_lon['lon_round2']==getRoundedThresholdv1(lon_this, 0.5)]['SPECIES_COUNT'].values
#%%
df['lat_round'] = df.apply(lambda row: getRoundedThresholdv1(row['LATITUDE'], 0.02), axis=1)
df['lon_round'] = df.apply(lambda row: getRoundedThresholdv1(row['LONGITUDE'], 0.02), axis=1)
#%%
print(df.mean())
#for chunk in df_reader:
#    print(chunk.groupby('SPECIES_COUNT').mean())

#%%
cl = pd.DataFrame.from_dict({ii: checklists[ii] for ii in checklists.keys()}, orient='index')

#%% Now try dask
df = dd.read_csv("test_ebird.txt", delimiter='\t', blocksize=2500)
df.compute()
#%%
checklists={}
group_identifier = {}
counter = 0
start = time.time()
for chunk in df_reader:
    end = time.time()
    print("chunk time: {:.3f}".format(end - start))
    start = time.time()
#    print(counter * 2000000)
    checklists, group_identifier = filterChunk(chunk, checklists, group_identifier)
    counter += 1
    end = time.time()
    print("filter time: {:.3f}".format(end - start))
    start = time.time()

#%%
chunk[(chunk['OBSERVATION DATE'] > '2010-01-01') & (chunk['OBSERVATION DATE'] < '2018-12-31')]
#%%
pure_df = df[df['PROTOCOL TYPE']!='Incidental']
#%%
pure_df = df[df['PROTOCOL TYPE']!='Incidental'] \
      [(~df.duplicated(subset='GROUP IDENTIFIER', keep='first')) | (df['GROUP IDENTIFIER'].isnull())]
print(pure_df)

#%%
vc = pure_df['SAMPLING EVENT IDENTIFIER'].value_counts().to_frame()
vc['value_counts'] = vc['SAMPLING EVENT IDENTIFIER']
vc['SAMPLING EVENT IDENTIFIER'] = list(vc.index)
count_df = pure_df.merge(vc, on='SAMPLING EVENT IDENTIFIER')
#%%
test_groupby = pure_df.groupby(['SAMPLING EVENT IDENTIFIER', 'LATITUDE', 'LONGITUDE', 'OBSERVATION DATE']).count()
#%%
cl_df = count_df.drop_duplicates(subset='SAMPLING EVENT IDENTIFIER', keep='first')
cl_df['lat_round'] = cl_df.apply(lambda row: getRoundedThresholdv1(row['LATITUDE'], 0.02))
cl_df['lon_round'] = cl_df.apply(lambda row: getRoundedThresholdv1(row['LONGITUDE'], 0.02))
#full_frame['class_id_sp'] = full_frame.apply(lambda row: corresponding_class[row.class_id], axis=1)