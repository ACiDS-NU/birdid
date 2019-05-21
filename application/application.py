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
import uuid
import csv
from werkzeug.utils import secure_filename
import pandas as pd
#import matplotlib.pyplot as plt
#import matplotlib.patches as patches
import tensorflow as tf
#from sklearn.model_selection import train_test_split
#from keras.utils import to_categorical
#from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
#from keras.preprocessing import image
keras = tf.keras
print(tf.__version__)

UPLOAD_FOLDER = 'upload/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
img_root = 'https://s3.us-east-2.amazonaws.com/plover-birdid/bird_img/'

def top3_idx(probs):
    return np.flip(np.argsort(probs)[-3:],0)

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
				except:
					x1 = 0
					x2 = im_w
					y1 = 0
					y2 = im_h
					w = im_w
					h = im_h

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
			t3_idx = top3_idx(probs)
			print(t3_idx)
			print("Prediction: {} ({:.1f} %), {} ({:.1f} %), or {} ({:.1f} %)".format(
				Birds[int(class_indices_inv_map[t3_idx[0]])], probs[t3_idx[0]]*100,
				Birds[int(class_indices_inv_map[t3_idx[1]])], probs[t3_idx[1]]*100,
				Birds[int(class_indices_inv_map[t3_idx[2]])], probs[t3_idx[2]]*100))
			b1, p1, b2, p2, b3, p3 = (Birds[int(class_indices_inv_map[t3_idx[0]])], '{:.1f}'.format(probs[t3_idx[0]]*100),
				Birds[int(class_indices_inv_map[t3_idx[1]])], '{:.1f}'.format(probs[t3_idx[1]]*100),
				Birds[int(class_indices_inv_map[t3_idx[2]])], '{:.1f}'.format(probs[t3_idx[2]]*100))
			BD1, BD2, BD3 = (Markup(Bird_description[b1]), Markup(Bird_description[b2]), Markup(Bird_description[b3]))
			# BI1 = 'static/bird_img/' + str(bird_img.loc[bird_img['class_name_sp'].isin([b1])].sample(n=1)['image_name_fname_only'].values[0])
			# BI2 = 'static/bird_img/' + str(bird_img.loc[bird_img['class_name_sp'].isin([b2])].sample(n=1)['image_name_fname_only'].values[0])
			# BI3 = 'static/bird_img/' + str(bird_img.loc[bird_img['class_name_sp'].isin([b3])].sample(n=1)['image_name_fname_only'].values[0])
			BI1 = bird_img.loc[bird_img['class_name_sp'].isin([b1])].sample(n=1)
			BI2 = bird_img.loc[bird_img['class_name_sp'].isin([b2])].sample(n=1)
			BI3 = bird_img.loc[bird_img['class_name_sp'].isin([b3])].sample(n=1)
			BIF1 = img_root + str(BI1['image_name_fname_only'].values[0])
			BIF2 = img_root + str(BI2['image_name_fname_only'].values[0])
			BIF3 = img_root + str(BI3['image_name_fname_only'].values[0])
			PH1 = str(BI1['photographer'].values[0])
			PH2 = str(BI2['photographer'].values[0])
			PH3 = str(BI3['photographer'].values[0])
			BL1 = 'https://en.wikipedia.org/wiki/' + Bird_link[b1]
			BL2 = 'https://en.wikipedia.org/wiki/' + Bird_link[b2]
			BL3 = 'https://en.wikipedia.org/wiki/' + Bird_link[b3]
			Bird1 = {'bird': b1, 'prob': p1, 'description': BD1, 'image': BIF1, 'bird_link': BL1, 'photographer': PH1}
			Bird2 = {'bird': b2, 'prob': p2, 'description': BD2, 'image': BIF2, 'bird_link': BL2, 'photographer': PH2}
			Bird3 = {'bird': b3, 'prob': p3, 'description': BD3, 'image': BIF3, 'bird_link': BL3, 'photographer': PH3}
			# print(os.path.join(application.config['UPLOAD_FOLDER'], filename))
			cv2.imwrite(os.path.join(application.config['UPLOAD_FOLDER'], filename), paint_to_square(img_r, desired_size=500, pad=False))
			# file.save(os.path.join(application.config['UPLOAD_FOLDER'], filename))
			# return render_template("home.html")#redirect(url_for('uploaded_file', filename=filename))
			return render_template("results.html", filename=filename, Bird1=Bird1, Bird2=Bird2, Bird3=Bird3)
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
