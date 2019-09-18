# -*- coding: utf-8 -*-
"""
Created on Mon May 27 20:57:20 2019

@author: user
"""

import pandas as pd
import numpy as np
import time
import datetime

def getRoundedThresholdv1(a, MinClip):
    return round(float(a) / MinClip) * MinClip

def filterChunk(chunk, checklists):
    chunk = chunk[chunk['COUNTRY CODE']=='US']
    chunk = chunk[chunk['PROTOCOL TYPE']!='Incidental']
    chunk = chunk[(chunk['OBSERVATION DATE'] > '2010-01-01') & (chunk['OBSERVATION DATE'] < '2018-12-31')]
    if len(chunk) > 0:
        print(len(chunk))
    for index, row in chunk.iterrows():
#        print(row)
#        print(type(row))
#        print(row['SAMPLING EVENT IDENTIFIER'])
        if row['SAMPLING EVENT IDENTIFIER'] in checklists:
            if checklists[row['SAMPLING EVENT IDENTIFIER']]['OBSERVER ID'] == row['OBSERVER ID']:
                checklists[row['SAMPLING EVENT IDENTIFIER']]['SPECIES COUNT'] += 1
        else:
            checklists[row['SAMPLING EVENT IDENTIFIER']] = {'OBSERVER ID': row['OBSERVER ID'],
                      'LATITUDE': row['LATITUDE'], 'LONGITUDE': row['LONGITUDE'], 'STATE': row['STATE'],
                      'COUNTY': row['COUNTY'], 'SPECIES COUNT': 1}
#            checklists[row['SAMPLING EVENT IDENTIFIER']] = {}
#            checklists[row['SAMPLING EVENT IDENTIFIER']]['OBSERVER ID'] = row['OBSERVER ID']
#            checklists[row['SAMPLING EVENT IDENTIFIER']]['LATITUDE'] = row['LATITUDE']
#            checklists[row['SAMPLING EVENT IDENTIFIER']]['LONGITUDE'] = row['LONGITUDE']
#            checklists[row['SAMPLING EVENT IDENTIFIER']]['STATE'] = row['STATE']
#            checklists[row['SAMPLING EVENT IDENTIFIER']]['COUNTY'] = row['COUNTY']
#            checklists[row['SAMPLING EVENT IDENTIFIER']]['SPECIES COUNT'] = 1
    return checklists
            

#%%
#df = pd.read_csv("test_ebird.txt", delimiter='\t')
#df_reader = pd.read_csv("E:\ebd_sampling_relApr-2019\ebd_relApr-2019.txt.gz", delimiter='\t', iterator=True, compression='gzip', chunksize=2000000)
df_reader = pd.read_csv("E:\ebd_sampling_relApr-2019\ebd_relApr-2019.txt", delimiter='\t', iterator=True, chunksize=2000000)
#df_reader = pd.read_csv("test_ebird.txt", delimiter='\t', iterator=True, chunksize=50000)
#%%
checklists={}
counter = 0
start = time.time()
for chunk in df_reader:
    end = time.time()
    print("chunk time: {:.3f}".format(end - start))
    start = time.time()
    print(counter * 2000000)
    checklists = filterChunk(chunk, checklists)
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