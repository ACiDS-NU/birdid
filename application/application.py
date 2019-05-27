# -*- coding: utf-8 -*-
"""
Created on Sat May 11 17:12:00 2019

@author: user
"""

import os
from flask import Flask, flash, redirect, render_template, \
request, url_for, send_from_directory, Markup
import numpy as np
import pickle
import cv2
import datetime
import json
import urllib.request
import uuid
import csv
from werkzeug.utils import secure_filename
import pandas as pd
import tensorflow as tf
keras = tf.keras
print(tf.__version__)

UPLOAD_FOLDER = 'upload/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
img_root = 'https://s3.us-east-2.amazonaws.com/plover-birdid/bird_img/'
#Find_taxon_occurrence = "http://api.gbif.org/v1/occurrence/search?country=US&dataset_key=4fa7b334-ce0d-4e88-aaae-2e0c138d049e&has_coordinate=true&has_geospatial_issue=false&taxon_key={TAXON_KEY}&event_date={X1},{X2}&event_date={X3},{X4}&event_date={X5},{X6}&geometry={GEOMETRY}&limit=0"
#Find_occurrence = "http://api.gbif.org/v1/occurrence/search?country=US&dataset_key=4fa7b334-ce0d-4e88-aaae-2e0c138d049e&has_coordinate=true&has_geospatial_issue=false&event_date={X1},{X2}&event_date={X3},{X4}&event_date={X5},{X6}&geometry={GEOMETRY}&limit=0"
Find_taxon_occurrence = "http://api.gbif.org/v1/occurrence/search?country=US&dataset_key=4fa7b334-ce0d-4e88-aaae-2e0c138d049e&has_coordinate=true&has_geospatial_issue=false&taxon_key={TAXON_KEY}&event_date={X1},{X2}&event_date={X3},{X4}&geometry={GEOMETRY}&limit=0"
Find_occurrence = "http://api.gbif.org/v1/occurrence/search?country=US&dataset_key=4fa7b334-ce0d-4e88-aaae-2e0c138d049e&has_coordinate=true&has_geospatial_issue=false&event_date={X1},{X2}&event_date={X3},{X4}&geometry={GEOMETRY}&limit=0"

def topn_idx(probs, n=3):
    return np.flip(np.argsort(probs)[-n:],0)

def more_than_some_percent(probs, thres=0.01):
	return np.argwhere(probs > thres)

def last_year(x, day_diff=-365):
	return x + datetime.timedelta(days=day_diff)

def time_span(x, span=7):
	# return plus minus <span> days
	return (x + datetime.timedelta(days=-span-365)).date(), (x + datetime.timedelta(days=span-365)).date()

def calc_poly(lat, lon, km=20):
	lat_deg = km / 2 / 111 # This is in km using length of 1 deg latitude ~= 111 km at 40 deg N
	lon_deg_u = lat_deg / np.cos((lat + lat_deg) * np.pi / 180.)
	lon_deg_d = lat_deg / np.cos((lat - lat_deg) * np.pi / 180.)
	print('POLYGON(({:.2f}%20{:.2f},{:.2f}%20{:.2f},{:.2f}%20{:.2f},{:.2f}%20{:.2f},{:.2f}%20{:.2f}))'.format(
		lon-lon_deg_u, lat+lat_deg, lon-lon_deg_d, lat-lat_deg, 
		lon+lon_deg_d, lat-lat_deg, lon+lon_deg_u, lat+lat_deg,
		lon-lon_deg_u, lat+lat_deg))
	return 'POLYGON(({:.2f}%20{:.2f},{:.2f}%20{:.2f},{:.2f}%20{:.2f},{:.2f}%20{:.2f},{:.2f}%20{:.2f}))'.format(
		lon-lon_deg_u, lat+lat_deg, lon-lon_deg_d, lat-lat_deg, 
		lon+lon_deg_d, lat-lat_deg, lon+lon_deg_u, lat+lat_deg,
		lon-lon_deg_u, lat+lat_deg,)
	

def _request_taxon_occurence(taxon_key, x1, x2, poly):
	x3 = last_year(x1)
	x4 = last_year(x2)
	x5 = last_year(x3)
	x6 = last_year(x4)
	# print(Find_taxon_occurrence.format(TAXON_KEY=taxon_key, X1=x1, X2=x2, X3=x3, X4=x4, X5=x5, X6=x6, GEOMETRY=poly))
	# with urllib.request.urlopen(Find_taxon_occurrence.format(TAXON_KEY=taxon_key, X1=x1, X2=x2, X3=x3, X4=x4, X5=x5, X6=x6, GEOMETRY=poly)) as req:
	print(Find_taxon_occurrence.format(TAXON_KEY=taxon_key, X1=x1, X2=x2, X3=x3, X4=x4, GEOMETRY=poly))
	with urllib.request.urlopen(Find_taxon_occurrence.format(TAXON_KEY=taxon_key, X1=x1, X2=x2, X3=x3, X4=x4, GEOMETRY=poly)) as req:

		data = req.read().decode("UTF-8")
	return data

def _request_occurrence(x1, x2, poly):
	x3 = last_year(x1)
	x4 = last_year(x2)
	x5 = last_year(x3)
	x6 = last_year(x4)
	# print(Find_occurrence.format(X1=x1, X2=x2, X3=x3, X4=x4, X5=x5, X6=x6, GEOMETRY=poly))
	# with urllib.request.urlopen(Find_occurrence.format(X1=x1, X2=x2, X3=x3, X4=x4, X5=x5, X6=x6, GEOMETRY=poly)) as req:
	print(Find_occurrence.format(X1=x1, X2=x2, X3=x3, X4=x4, GEOMETRY=poly))
	with urllib.request.urlopen(Find_occurrence.format(X1=x1, X2=x2, X3=x3, X4=x4, GEOMETRY=poly)) as req:
		data = req.read().decode("UTF-8")
	return data

def paint_to_square(img, desired_size=224, pad=True):
    old_size = img.shape[:2] # old_size is in (height, width) format
    print(old_size)
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format

    im = cv2.resize(img, (new_size[1], new_size[0]))
    if pad:
	    delta_w = desired_size - new_size[1]
	    delta_h = desired_size - new_size[0]
	    top, bottom = delta_h//2, delta_h-(delta_h//2)
	    left, right = delta_w//2, delta_w-(delta_w//2)
	    
	    color = [0, 0, 0]
	    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
	        value=color)#new_im = cv2.cvtColor(new_im, cv2.COLOR_BGR2RGB)
    else:
        return im

    return new_im


application = Flask(__name__)
application.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
application.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

def allowed_file(filename):
	return '.' in filename and \
	filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# @application.route('/')
# def index():
# 	return render_template('index.html')

@application.route('/about')
def about():
	return render_template('about.html')

@application.route('/how_it_works')
def how_it_works():
	return render_template('how_it_works.html')

@application.route('/', methods=['GET', 'POST'])
def index():
	if request.method == 'POST':
		messages = {}
		# check if the post request has the file part
		if 'file' not in request.files:
			# flash('No file part')
			return redirect(request.url)
		file = request.files['file']
		# filestr = file.read()
		# if user does not select file, browser also
		# submit a empty part without filename
		if file.filename == '':
		# flash('No selected file')
			return redirect(request.url)

		try:
			obs_date = datetime.datetime.strptime(request.form['date'], '%Y-%m-%d')
			messages["obs_date_default"] = 0
			messages["obs_date"] = obs_date.date()
			Do_GeoSpatial_Filtering_date = 1
		except:
			obs_date = datetime.datetime.now()
			messages["obs_date_default"] = 1
			Do_GeoSpatial_Filtering_date = 1
			print("Date defaulted.")
		try:
			lat = float(request.form['lat'])
			lon = float(request.form['lon'])
			messages['location'] = request.form['location']
			print(lat, lon)
			poly = calc_poly(lat, lon, km=20)
			print(poly)
			Do_GeoSpatial_Filtering_latlon = 1
			messages["lat_lon_default"] = 0
			messages["lat"] = '{:.2f}'.format(lat)
			messages["lon"] = '{:.2f}'.format(lon)

		except:
			Do_GeoSpatial_Filtering_latlon = 0
			messages["lat_lon_default"] = 1
			print("Lat lon defaulted.")

		if Do_GeoSpatial_Filtering_latlon and Do_GeoSpatial_Filtering_date:
			GeoSpatial_Filtering = 1
			messages["GeoSpatial_Filtering"] = 1
			print("We are doing GeoSpatial_Filtering")
		else:
			GeoSpatial_Filtering = 0
			messages["GeoSpatial_Filtering"] = 0
			print("We are not doing GeoSpatial_Filtering")


		print(obs_date.date())
		if file and allowed_file(file.filename):
			file_ext = '.' + secure_filename(file.filename).rsplit('.', 1)[1].lower()
			filename = str(uuid.uuid4()) + file_ext
			filestr = file.read()
			npimg = np.fromstring(filestr, np.uint8)
			img_r = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
			img = cv2.cvtColor(img_r , cv2.COLOR_BGR2RGB)
			# img = img_r

			im_h = img.shape[0]
			im_w = img.shape[1]
			print(im_h, im_w)

			if all (k in request.form for k in ("x1","x2","y1","y2","w","h")):
				try:
					x1 = int(request.form['x1'])
					x2 = int(request.form['x2'])
					y1 = int(request.form['y1'])
					y2 = int(request.form['y2'])
					w = int(request.form['w'])
					h = int(request.form['h'])
					messages["im_selection_default"] = 0
				except:
					x1 = 0
					x2 = im_w
					y1 = 0
					y2 = im_h
					w = im_w
					h = im_h
					messages["im_selection_default"] = 1

			print(x1, x2, y1, y2, w, h)
			actual_x1 = round(x1 / w * im_w)
			actual_x2 = round(x2 / w * im_w)
			actual_y1 = round(y1 / h * im_h)
			actual_y2 = round(y2 / h * im_h)
			if (actual_x2 - actual_x1) == 0 or (actual_y2 - actual_y1) == 0:
				actual_x1 = 0
				actual_x2 = im_w
				actual_y1 = 0
				actual_y2 = im_h
				messages["im_selection_default"] = 1

			# if actual_h > actual_w:
			# 	actual_y2 -= actual_h - actual_w
			# if actual_w > actual_h:
			# 	actual_x2 -= actual_w - actual_h
			# print(type(img))
			img = img[actual_y1:actual_y2,actual_x1:actual_x2]
			img_r = img_r[actual_y1:actual_y2,actual_x1:actual_x2]
			img = paint_to_square(img)
			data = np.expand_dims(img, axis = 0) / 255.0
			# print(data.shape)
			probs = model.predict(data,verbose=1).flatten()
			# print(probs)
			# print(probs.shape)
			occ = {}
			if GeoSpatial_Filtering:
				
				prob_idx = more_than_some_percent(probs).flatten()
				print(prob_idx)
				print(probs[prob_idx])
				#today = datetime.datetime.now()+datetime.timedelta(days=-60)
				x1, x2 = time_span(obs_date, span=7)
				occ['total'] = json.loads(_request_occurrence(x1, x2, poly))['count']
				for idx, ii in enumerate(prob_idx):
					# Gather geo-spatial info
					Bird_this = Birds[int(class_indices_inv_map[ii])]
					taxon_this = Bird_taxon[Bird_this]
					taxon_occurrence_this = json.loads(_request_taxon_occurence(taxon_this, x1, x2, poly))['count']
					occ[ii] = taxon_occurrence_this
					if taxon_occurrence_this / occ['total'] < 0.0003: # Temporary
						probs[ii] = 0
					# taxon_occurrence_AR = json.loads(_request_taxon_occurence(9510564, x1, x2)) # American Robin
				# tn_idx = prob_idx[np.argsort(np.multiply(probs[prob_idx], taxon_occurrence),0)[0:3]] # ish
				tn_idx = topn_idx(probs, n=5)
				tn_idx = tn_idx[np.isin(tn_idx, prob_idx)]
			else:
				occ['total'] = 0
				tn_idx = topn_idx(probs, n=3)
			print(tn_idx)
			# print(today)
			# print("Prediction: {} ({:.1f} %), {} ({:.1f} %), or {} ({:.1f} %)".format(
			# 	Birds[int(class_indices_inv_map[tn_idx[0]])], probs[tn_idx[0]]*100,
			# 	Birds[int(class_indices_inv_map[tn_idx[1]])], probs[tn_idx[1]]*100,
			# 	Birds[int(class_indices_inv_map[tn_idx[2]])], probs[tn_idx[2]]*100))
			Bird_candidates = []
			for ii in range(len(tn_idx)):
				b, p = (Birds[int(class_indices_inv_map[tn_idx[ii]])], '{:.1f}'.format(probs[tn_idx[ii]]*100))
				BD = Markup(Bird_description[b])
				BI = bird_img.loc[bird_img['class_name_sp'].isin([b])].sample(n=1)
				BIF = img_root + str(BI['image_name_fname_only'].values[0])
				PH = str(BI['photographer'].values[0])
				BL = 'https://en.wikipedia.org/wiki/' + Bird_link[b]
				if not GeoSpatial_Filtering:
					occ[tn_idx[ii]] = 0
				Bird = {'bird': b, 'prob': p, 'description': BD, 'image': BIF, 'bird_link': BL, 'photographer': PH, 'occ': occ[tn_idx[ii]]}
				Bird_candidates.append(Bird)
			# if len(tn_idx) > 0:
			# 	b1, p1 = (Birds[int(class_indices_inv_map[tn_idx[0]])], '{:.1f}'.format(probs[tn_idx[0]]*100))
			# 	BD1 = Markup(Bird_description[b1])
			# 	BI1 = bird_img.loc[bird_img['class_name_sp'].isin([b1])].sample(n=1)
			# 	BIF1 = img_root + str(BI1['image_name_fname_only'].values[0])
			# 	PH1 = str(BI1['photographer'].values[0])
			# 	BL1 = 'https://en.wikipedia.org/wiki/' + Bird_link[b1]
			# 	if not GeoSpatial_Filtering:
			# 		occ[tn_idx[0]] = 0
			# 	Bird1 = {'bird': b1, 'prob': p1, 'description': BD1, 'image': BIF1, 'bird_link': BL1, 'photographer': PH1, 'occ': occ[tn_idx[0]]}
				
			# else:
			# 	Bird1 = []
			# if len(tn_idx) > 1:
			# 	b2, p2 = (Birds[int(class_indices_inv_map[tn_idx[1]])], '{:.1f}'.format(probs[tn_idx[1]]*100))
			# 	BD2 = Markup(Bird_description[b2])
			# 	BI2 = bird_img.loc[bird_img['class_name_sp'].isin([b2])].sample(n=1)
			# 	BIF2 = img_root + str(BI2['image_name_fname_only'].values[0])
			# 	PH2 = str(BI2['photographer'].values[0])
			# 	BL2 = 'https://en.wikipedia.org/wiki/' + Bird_link[b2]
			# 	if not GeoSpatial_Filtering:
			# 		occ[tn_idx[1]] = 0
			# 	Bird2 = {'bird': b2, 'prob': p2, 'description': BD2, 'image': BIF2, 'bird_link': BL2, 'photographer': PH2, 'occ': occ[tn_idx[1]]}
				
			# else:
			# 	Bird2 = []
			# if len(tn_idx) > 2:
			# 	b3, p3 = (Birds[int(class_indices_inv_map[tn_idx[2]])], '{:.1f}'.format(probs[tn_idx[2]]*100))
			# 	BD3 = Markup(Bird_description[b3])
			# 	BI3 = bird_img.loc[bird_img['class_name_sp'].isin([b3])].sample(n=1)
			# 	BIF3 = img_root + str(BI3['image_name_fname_only'].values[0])
			# 	PH3 = str(BI3['photographer'].values[0])
			# 	BL3 = 'https://en.wikipedia.org/wiki/' + Bird_link[b3]
			# 	if not GeoSpatial_Filtering:
			# 		occ[tn_idx[2]] = 0
			# 	Bird3 = {'bird': b3, 'prob': p3, 'description': BD3, 'image': BIF3, 'bird_link': BL3, 'photographer': PH3, 'occ': occ[tn_idx[2]]}
				
			# else:
			# 	Bird3 = []

			# BI1 = 'static/bird_img/' + str(bird_img.loc[bird_img['class_name_sp'].isin([b1])].sample(n=1)['image_name_fname_only'].values[0])
			# BI2 = 'static/bird_img/' + str(bird_img.loc[bird_img['class_name_sp'].isin([b2])].sample(n=1)['image_name_fname_only'].values[0])
			# BI3 = 'static/bird_img/' + str(bird_img.loc[bird_img['class_name_sp'].isin([b3])].sample(n=1)['image_name_fname_only'].values[0])
			
			# print(os.path.join(application.config['UPLOAD_FOLDER'], filename))
			cv2.imwrite(os.path.join(application.config['UPLOAD_FOLDER'], filename), paint_to_square(img_r, desired_size=500, pad=False))
			# file.save(os.path.join(application.config['UPLOAD_FOLDER'], filename))
			# return render_template("home.html")#redirect(url_for('uploaded_file', filename=filename))
			# return render_template("results.html", filename=filename, Bird1=Bird1, Bird2=Bird2, Bird3=Bird3, num_birds=len(tn_idx), occ_tot=occ['total'], messages=messages)
			return render_template("results.html", filename=filename, Bird_candidates=Bird_candidates, num_birds=len(tn_idx), occ_tot=occ['total'], messages=messages)
	# return render_template("home.html")
	return render_template("index.html")

@application.route('/uploads/<filename>')
def uploaded_file(filename):
	return send_from_directory(application.config['UPLOAD_FOLDER'],filename)

# @application.route('/results/<filename>')
# def results(filename):
	

pkl_file = open('static/Birds.pkl', 'rb')
Birds = pickle.load(pkl_file)
pkl_file.close()
pkl_file = open('static/Bird_taxon.pkl', 'rb')
Bird_taxon = pickle.load(pkl_file)
pkl_file.close()
pkl_file = open('static/class_indices_inv_map.pkl', 'rb')
class_indices_inv_map = pickle.load(pkl_file)
pkl_file.close()
pkl_file = open('static/Bird_description_wikipedia.pkl', 'rb')
Bird_description = pickle.load(pkl_file)
pkl_file.close()
pkl_file = open('static/Bird_link.pkl', 'rb')
Bird_link = pickle.load(pkl_file)
pkl_file.close()
bird_img = pd.read_csv('static/bird_img.csv')
# Bird_description = dict(zip(Bird_description2a,Bird_description2b))
# print(type(Bird_description2))
# print(Bird_description2)
model = keras.models.load_model('static/model3_30.h5')

if __name__ == "__main__":
	application.run(debug=True)
