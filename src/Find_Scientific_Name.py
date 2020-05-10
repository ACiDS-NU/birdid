# -*- coding: utf-8 -*-
"""
Created on Tue May 21 21:30:14 2019

@author: user
"""

Find_taxon_key = "http://api.gbif.org/v1/species/search?q={SPECIES}&rank=SPECIES&limit=300"
Find_taxon_occurrence = "http://api.gbif.org/v1/occurrence/search?country=US&dataset_key=4fa7b334-ce0d-4e88-aaae-2e0c138d049e&has_coordinate=true&has_geospatial_issue=false&taxon_key={TAXON_KEY}&event_date={X1},{X2}&geometry=POLYGON((-87.84%2041.94,-87.60%2041.94,-87.60%2042.12,-87.84%2042.12,-87.84%2041.94))&limit=0"
Find_occurrence = "http://api.gbif.org/v1/occurrence/search?country=US&dataset_key=4fa7b334-ce0d-4e88-aaae-2e0c138d049e&has_coordinate=true&has_geospatial_issue=false&event_date={X1},{X2}&geometry=POLYGON((-87.84%2041.94,-87.60%2041.94,-87.60%2042.12,-87.84%2042.12,-87.84%2041.94))&limit={LIMIT}"
Find_all_occurrence = "http://api.gbif.org/v1/occurrence/search?country=US&dataset_key=4fa7b334-ce0d-4e88-aaae-2e0c138d049e&has_coordinate=true&has_geospatial_issue=false&taxon_key={TAXON_KEY}&limit=0"
Find_cata_occurrence = "http://api.gbif.org/v1/occurrence/search?country=US&catalogNumber={CATA}&dataset_key=4fa7b334-ce0d-4e88-aaae-2e0c138d049e&has_coordinate=true&has_geospatial_issue=false&limit=0"
import json
import urllib.request
import pickle
from bs4 import BeautifulSoup
import datetime
import matplotlib.pyplot as plt
import time


def _request_taxon_occurence(taxon_key, x1, x2):
    print(Find_taxon_occurrence.format(TAXON_KEY=taxon_key, X1=x1, X2=x2))
    with urllib.request.urlopen(Find_taxon_occurrence.format(TAXON_KEY=taxon_key, X1=x1, X2=x2)) as req:
        data = req.read().decode("UTF-8")
    return data


def _request_occurrence(x1, x2, limit=0):
    with urllib.request.urlopen(Find_occurrence.format(X1=x1, X2=x2, LIMIT=limit)) as req:
        data = req.read().decode("UTF-8")
    return data

def _request_all_occurrence(taxon_key):
    with urllib.request.urlopen(Find_all_occurrence.format(TAXON_KEY=taxon_key)) as req:
        data = req.read().decode("UTF-8")
    return data


def _request_taxon_key(name):
    with urllib.request.urlopen(Find_taxon_key.format(SPECIES=name)) as req:
        data = req.read().decode("UTF-8")
    return data

def _request_cata_occurrence(cata):
    with urllib.request.urlopen(Find_cata_occurrence.format(CATA=cata)) as req:
        data = req.read().decode("UTF-8")
    return data
#%%
pkl_file = open('../application/static/Bird_description_wikipedia.pkl', 'rb')
Bird_description = pickle.load(pkl_file)
pkl_file.close()

Bird_sci_name = {}
#%%
for name, desc in Bird_description.items():
    soup = BeautifulSoup(desc, 'html.parser')
    sci_name_this = soup.find_all("i")[0].text
    Bird_sci_name[name]= str(sci_name_this)
    
    
#%%
Bird_taxon = {}
problem_birds = []
#%%
for name, sci_name in Bird_sci_name.items():
    if name in Bird_taxon:
        pass
    else:
        Bird_this = sci_name.replace(" ","%20")
        j = json.loads(_request_taxon_key(name=Bird_this))['results']
        tried_taxon = []
        for ii in np.arange(len(j)):
            if 'nubKey' in j[ii]:
                if j[ii]['nubKey'] not in tried_taxon:
                    tried_taxon.append(j[ii]['nubKey'])
                    k = json.loads(_request_all_occurrence(j[ii]['nubKey']))
                    
                    if k['count'] > 200:
                        Bird_taxon[name] = j[ii]['nubKey']
                        Bird_count[name] = k['count']
                        print(name, j[ii]['nubKey'], k['count'])
                        break;
                    else:
#                        Bird_taxon[name] = j[ii]['nubKey']
                        print(name, j[ii]['nubKey'], k['count'], "doesn't work!")
        if name not in Bird_taxon:
            problem_birds.append(name)
            print("We have a problem with {}".format(name))
#        print(name, Bird_taxon[name])
        time.sleep(0.3)        
#%%
Bird_taxon['Vermilion_Flycatcher'] = 2483647
Bird_taxon['Northern_Pygmy-Owl'] = 9616953
Bird_taxon['Hairy_Woodpecker'] = 9149595 
# 9149595 is actually the taxon for Downy Woodpecker, 
# but in GBIF EOD Hairy is classified as unknown species so I had no choice.
#%%
Bird_count = {}
problem_birds = []
#%% Sanity check

for name, taxon in Bird_taxon.items():
    if name in Bird_count:
        pass
    else:
        j = json.loads(_request_all_occurrence(taxon))
        
        Bird_count[name] = j['count']
        if j['count'] < 200:
            problem_birds.append(name)
            print(name, j['count'], "problematic!!")
        else:
            print(name, j['count'])
#%%
f = open("Bird_taxon.pkl","wb")
pickle.dump(Bird_taxon,f)
f.close()
#%%
f = open("Bird_taxon.pkl","rb")
Bird_taxon = pickle.load(f)
f.close()
#%%
Bird_taxon['Red-winged_Blackbird'] = 9409198
#%%
Bird_this = Bird_sci_name['Blackburnian_Warbler'].replace(" ","%20")
taxon_this = json.loads(_request_taxon_key(name=Bird_this))['results'][0]['nubKey']
#taxon_occurrence = json.loads(_request_taxon_occurence(taxon_this))
taxon_occurrence = json.loads(_request_taxon_occurence(9510564)) # American Robin
#taxon_occurrence = json.loads(_request_taxon_occurence(2490384)) # Northern cardinal
total_occurrence = json.loads(_request_occurrence(limit=0))
#total_occurrence = json.loads(_request_taxon_occurence(9510564)) # American Robin
#%%
print(taxon_occurrence['count'])
print(total_occurrence['count'])
print("Percentage is {:.1f} %".format(taxon_occurrence['count']/total_occurrence['count']*100))
#%%

print(total_occurrence)


#%%
abundance = []
x = datetime.datetime(2017,1,1)
while x < datetime.datetime(2018,1,1):
    x1 = x.date()
    x2 = (x + datetime.timedelta(days=6)).date()
    taxon_occurrence = json.loads(_request_taxon_occurence(taxon_this, x1, x2)) # Blackburnian Warbler
#    taxon_occurrence_AR = json.loads(_request_taxon_occurence(9510564, x1, x2)) # American Robin
    total_occurrence = json.loads(_request_occurrence(x1, x2))
#    print(taxon_occurrence['count'])
#    print(total_occurrence['count'])
    print("Week is {X}, Percentage is {Y:.1f} %".format(X=x.date(),Y=taxon_occurrence['count']/total_occurrence['count']*100))
    x += datetime.timedelta(days=7)
    abundance.append(taxon_occurrence['count']/total_occurrence['count']*100)

plt.figure(figsize=(18,12))
plt.plot(abundance)
plt.show()
#%%
start_time = time.time()

Bird_this = 'Blackburnian_Warbler'
taxon_this = Bird_taxon[Bird_this]
today = datetime.datetime.now()+datetime.timedelta(days=1)
x1 = (today + datetime.timedelta(days=-3-365)).date()
x2 = (today + datetime.timedelta(days=3-365)).date()
taxon_occurrence = json.loads(_request_taxon_occurence(taxon_this, x1, x2)) # Blackburnian Warbler
taxon_occurrence_AR = json.loads(_request_taxon_occurence(9510564, x1, x2)) # American Robin
#total_occurrence = json.loads(_request_occurrence(x1, x2))

print("Week is {X}, Percentage is {Y:.1f} %".format(
        X=(today+datetime.timedelta(days=-365)).date(),Y=taxon_occurrence['count']/taxon_occurrence_AR['count']*100))

elapsed_time = time.time() - start_time
print(elapsed_time)
#%%
total_occurrence = json.loads(_request_occurrence(x1, x2, limit=300))
#%%
catalogList = set()
for ii in total_occurrence['results']:
    catalogList.add(ii['catalogNumber'])
    
print(catalogList)
#%%
for ii in catalogList:
    occ = json.loads(_request_cata_occurrence(ii))
    print(occ)