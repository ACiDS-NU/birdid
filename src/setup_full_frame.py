# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 10:33:52 2019

This is to build a full dataframe 

@author: djhsu
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from shutil import copyfile
import os


#%%
#def split_index(y, num_classes, savepath):
#
#
#    I_train, I_test, y_train, y_test = train_test_split(np.arange(len(y)), y, random_state=2200)
#
#    # X_train = X_train.astype('float32')
#    # X_test = X_test.astype('float32')
#
#    # X_test, X_val = X_test[:-6000], X_test[-6000:]
#    y_test, y_val = y_test[:-(len(y_test)//2)], y_test[-(len(y_test)//2):]
#    I_test, I_val = I_test[:-(len(I_test)//2)], I_test[-(len(I_test)//2):]
#    # convert class vectors to binary class matrices
#    Y_train = to_categorical(y_train, num_classes)
#    Y_val = to_categorical(y_val, num_classes)
#    Y_test = to_categorical(y_test, num_classes)
#    np.save(savepath + 'Y_train_' + str(num_classes)+ '.npy',Y_train)
#    np.save(savepath + 'Y_val_' + str(num_classes)+ '.npy',Y_val)
#    np.save(savepath + 'Y_test_' + str(num_classes)+ '.npy',Y_test)
#    np.save(savepath + 'y_train_' + str(num_classes)+ '.npy',y_train)
#    np.save(savepath + 'y_val_' + str(num_classes)+ '.npy',y_val)
#    np.save(savepath + 'y_test_' + str(num_classes)+ '.npy',y_test)
#
#    print('\nI_train shape: {}\n I_val shape: {}\n I_test shape: {}\n y_train shape: {}\n y_val shape: {}\n y_test shape: {}'.format(I_train.shape, I_val.shape, I_test.shape, y_train.shape, y_val.shape, y_test.shape))
#    #print('\nX_train shape: {}\n y_train shape: {}\n X_test shape: {}\n y_test shape: {}'.format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))
#    #print('Y_train shape: {}\n Y_test shape: {}'.format(Y_train.shape, Y_test.shape))
#
#    return I_train, I_val, I_test, y_train, y_val, y_test

def make_npy_many(df, bird_list, y_path, X_path, L_path):
    desired_size = 224
    channels = 3
    # y_path = ground truth label
    # X_path = data
    print(y_path)
#     df = df[(df['class_name']==bird1) | (df['class_name']==bird2)]
    df = df[df['class_name'].isin(bird_list)]
    df['class_id_2'] = pd.Series(np.zeros(len(df)), index=df.index)
#     print(df)
    var = df.class_id.unique()
    print(var)
    for i, z in enumerate(var):
        df.loc[df['class_id']==z,'class_id_2'] = i
#         print(df[df['class_id']==z])
#     print(df)
        print('Now processing bird class = {}: {} '.format(i,df.loc[df['class_id']==z,'class_name'].iloc[0]))
#     print('Bird class = 0: ', df.iloc[-1]['image_name'])
    target = df['class_id_2'].values.astype(int)
    np.save(y_path, target)
#     print(target)
    zipper = zip(df.image_name, df.bb_y, df.bb_height, df.bb_x, df.bb_width)
    #new_x = np.zeros((len(df),desired_size, desired_size, channels), dtype=np.uint8)
    print(new_x.shape)
    print(new_x.size)
    for i, z in enumerate(zipper):
        #print(datapath + '/images/' + z[0])
        image = cv2.imread('../../images/' + z[0])
        cropped = image[z[1]:(z[1]+z[2]), z[3]:(z[3]+z[4])]
        #new_x.append(paint_to_square(cropped, z).astype(np.uint8))
        #new_x[i,:,:,:] = paint_to_square(cropped, z).astype(np.uint8)
        if i % 100 == 0:
            print("Processing image {}.".format(i))
        if (i < 10):
            print(image.shape)
            print(cropped.shape)
            cropped = paint_to_square(cropped, z)
            print(cropped.shape)
            fig,ax = plt.subplots(1)
            #plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            plt.imshow(cropped)
            plt.show()

    #X = np.array(new_x)
    np.save(X_path, new_x)
    
    np.save(L_path, bird_list)
    
    #return target, new_x


def paint_to_square(img, z):
    desired_size = 400
    old_size = img.shape[:2] # old_size is in (height, width) format

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format

    im = cv2.resize(img, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    
    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)
    #new_im = cv2.cvtColor(new_im, cv2.COLOR_BGR2RGB)
    return new_im

def paint_to_square_no_crop(img, ds, verbose=0):

    return img
#    s = img.shape[:2]
#    if verbose:
#        print("In paint_to_square_no_crop",s[0], s[1])
#    
#    delta_w = ds - s[1]
#    delta_h = ds - s[0]
#    top, bottom = delta_h//2, delta_h-(delta_h//2)
#    left, right = delta_w//2, delta_w-(delta_w//2)
#    
#    color = [0, 0, 0]
#    new_im = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
#        value=color)
#    #new_im = cv2.cvtColor(new_im, cv2.COLOR_BGR2RGB)
#    return delta_w//2, delta_h//2, new_im


def square_bbox(image, z, verbose=0):
    # Simply crop the image to a square box using maximal possible area
    # Pic attributes
    (im_height, im_width, _) = image.shape
    if verbose:
        print("In square_bbox",im_height, im_width)
    r_height = im_height - z['bb_y'] - z['bb_height']
    r_width = im_width - z['bb_x'] - z['bb_width']
    if verbose:
        print(z['bb_y'], z['bb_height'],r_height, z['bb_x'], z['bb_width'],r_width)
    if z['bb_width'] > z['bb_height']:
        nbsl = (int)(round(z['bb_width'] * 1.1,0)) # New bounding box side length
    else:
        nbsl = (int)(round(z['bb_height'] * 1.1,0))

    if verbose:    
        print(nbsl)
    
    d_width = (nbsl - z['bb_width']) // 2
    d_height = (nbsl - z['bb_height']) // 2
    if verbose:
        print(d_height, d_width)

    # Pad
    delta_h = 0
    delta_w = 0

    if z['bb_width'] > z['bb_height']:
        if verbose:
            print('Bbox width larger than height')
        # Width part
        if nbsl > im_width:
            if verbose:
                print('Expanded bbox larger than the pic width')
            bx0 = 0
            bx1 = im_width
            dx0 = 0
            nbsl = im_width
            d_width = (nbsl - z['bb_width']) // 2
            d_height = (nbsl - z['bb_height']) // 2
        elif (z['bb_x'] < d_width):
            if verbose:
                print('Left x too small')
            bx0 = 0
            bx1 = nbsl
            dx0 = d_width - z['bb_x']
        elif (r_width < d_width):
            if verbose:
                print('Right x too small')
            bx0 = im_width - nbsl
            bx1 = im_width
            dx0 = d_width - r_width
        else:
            if verbose:
                print('Expanding x')
            bx0 = z['bb_x'] - d_width
            bx1 = z['bb_x'] + z['bb_width'] + d_width
            dx0 = 0
        # Height part
        if nbsl > im_height:
            if verbose:
                print('Expanded bbox larger than the pic height 1')
            delta_w, delta_h, image = paint_to_square_no_crop(image[0:im_height, bx0:bx1], nbsl)
            by0 = 0
            by1 = nbsl
            bx0 = 0
            bx1 = nbsl
            rect = patches.Rectangle((d_width-dx0,z['bb_y']+delta_h),z['bb_width'],z['bb_height'],linewidth=1,edgecolor='r',facecolor='none')

        elif (z['bb_y'] < d_height) & (r_height < d_height):
            if verbose:
                print('Expanded bbox larger than the pic height 2')
            delta_w, delta_h, image = paint_to_square_no_crop(image[0:im_height, bx0:bx1], nbsl)
            by0 = 0
            by1 = nbsl
            bx0 = 0
            bx1 = nbsl
            rect = patches.Rectangle((d_width-dx0,z['bb_y']+delta_h),z['bb_width'],z['bb_height'],linewidth=1,edgecolor='r',facecolor='none')
            
        elif (z['bb_y'] < d_height):
            if verbose:
                print('Low y too small')
            by0 = 0
            by1 = nbsl
            rect = patches.Rectangle((z['bb_x']-dx0,z['bb_y']),z['bb_width'],z['bb_height'],linewidth=1,edgecolor='r',facecolor='none')
        elif (r_height < d_height):
            if verbose:
                print('High y too small')
            by0 = im_height - nbsl
            by1 = im_height
            rect = patches.Rectangle((z['bb_x']-dx0,z['bb_y']),z['bb_width'],z['bb_height'],linewidth=1,edgecolor='r',facecolor='none')
        else:
            if verbose:
                print('Expanding y')
            by0 = z['bb_y'] - d_height
            by1 = z['bb_y'] + z['bb_height'] + d_height
            rect = patches.Rectangle((z['bb_x']-dx0,z['bb_y']),z['bb_width'],z['bb_height'],linewidth=1,edgecolor='r',facecolor='none')

    if z['bb_height'] >= z['bb_width']:
        if verbose:
            print('Bbox height larger than width (or same)')
        # Height part
        if nbsl > im_height:
            if verbose:
                print('Expanded bbox larger than the pic height')
            by0 = 0
            by1 = im_height
            dy0 = 0
            nbsl = im_height
            d_width = (nbsl - z['bb_width']) // 2
            d_height = (nbsl - z['bb_height']) // 2
        elif (z['bb_y'] < d_height):
            
            by0 = 0
            by1 = nbsl
            dy0 = d_height - z['bb_y']
        elif (r_height < d_height):
            by0 = im_height - nbsl
            by1 = im_height
            dy0 = d_height - r_height
        else:
            by0 = z['bb_y'] - d_height
            by1 = z['bb_y'] + z['bb_height'] + d_height
            dy0 = 0
        # Width part
        if nbsl > im_width:
            delta_w, delta_h, image = paint_to_square_no_crop(image[by0:by1,0:im_width], nbsl)
            bx0 = 0
            bx1 = nbsl
            by0 = 0
            by1 = nbsl
            rect = patches.Rectangle((z['bb_x']+delta_w,d_height-dy0),z['bb_width'],z['bb_height'],linewidth=1,edgecolor='r',facecolor='none')
            
        elif (z['bb_x'] < d_width) & (r_width < d_width):
            #print("Both sides not sufficient")
            delta_w, delta_h, image = paint_to_square_no_crop(image[by0:by1,0:im_width], nbsl)
            bx0 = 0
            bx1 = nbsl
            by0 = 0
            by1 = nbsl
            rect = patches.Rectangle((z['bb_x']+delta_w,d_height-dy0),z['bb_width'],z['bb_height'],linewidth=1,edgecolor='r',facecolor='none')

        elif (z['bb_x'] < d_width):
            bx0 = 0
            bx1 = nbsl
            rect = patches.Rectangle((z['bb_x'],z['bb_y']-dy0),z['bb_width'],z['bb_height'],linewidth=1,edgecolor='r',facecolor='none')
        elif (r_width < d_width):
            bx0 = im_width - nbsl
            bx1 = im_width
            rect = patches.Rectangle((z['bb_x'],z['bb_y']-dy0),z['bb_width'],z['bb_height'],linewidth=1,edgecolor='r',facecolor='none')
        else:
            bx0 = z['bb_x'] - d_width
            bx1 = z['bb_x'] + z['bb_width'] + d_width
            rect = patches.Rectangle((z['bb_x'],z['bb_y']-dy0),z['bb_width'],z['bb_height'],linewidth=1,edgecolor='r',facecolor='none')
    
    return by0, by1, bx0, bx1, delta_h, delta_w, nbsl, image, rect



#%%
if __name__ == '__main__':
    # Read files
    file_dir = '../nabirds/nabirds/'
    images = pd.read_csv(file_dir + 'images.txt', sep=" ", header=None, names = ["image_id","image_name"])
    print(images)
    #imagesFrame = pd.DataFrame([images], columns = )
    print("read image")
    train_test_split = pd.read_csv(file_dir + 'train_test_split.txt', sep=" ", header=None, names = ["image_id","is_training"])
    #trainTestSplitFrame = pd.DataFrame([train_test_split],columns = )
    print("read split")
    image_size = pd.read_csv(file_dir + 'sizes.txt', sep=" ", header=None, names = ["image_id","image_width","image_height"])
    #imageSizeFrame = pd.DataFrame([bounding_box],column )
    print("read image size")
    classes = pd.read_csv(file_dir + 'classes2.txt', sep=" ", header=None, names = ["class_id", "class_name"])
    #classesFrame = pd.DataFrame([classes],columns)
    print("read classes")
    image_class_label = pd.read_csv(file_dir + 'image_class_labels.txt', sep=" ", header=None, names = ["image_id","class_id"])
    #imageClassLabelFrame = pd.DataFrame([image_class_label],columns)
    print("read image class label")
    hierarchy = pd.read_csv(file_dir + 'hierarchy.txt', sep=" ", header=None, names = ["child_class_id","parent_class_id"])
    #hierarchyFrame = pd.DataFrame([hierarchy],columns)
    print("read hierarchy")    
    bounding_box = pd.read_csv(file_dir + 'bounding_boxes.txt', sep=" ", header=None, names = ["image_id","bb_x","bb_y","bb_width","bb_height"])
    #boundingBoxFrame = pd.DataFrame([bounding_box],column = ["image_id","bb_x","bb_y","bb_width","bb_height"])
    print("read bounding boxes")
    photographer = pd.read_csv(file_dir + 'photographers2.txt',sep=" ", header=None, names = ["image_id", "photographer"])
    print("read photographer")
    #print(classes)
    
    # Concatenate to a full frame including
    # ['image name','class id','x','y','x_end ( = x + x_dim)','y_end']
    # 
    
    img_bbox = pd.merge(images, bounding_box, on="image_id")
    img_class_bbox = pd.merge(img_bbox,image_class_label,on="image_id")
    img_class_bbox_photo = pd.merge(img_class_bbox, photographer, on="image_id")
    full_frame = pd.merge(img_class_bbox_photo,classes,on="class_id")
    full_frame['image_file'] = full_frame['image_id'] + '.png'
    full_frame
    
    #%%
if __name__ == '__main__':
#    for i in np.arange(402,10000,2000):
    for i in [9382]:
        print(i)
        z = full_frame.iloc[i]
        print(z['class_name_sp'])
        print(z['image_name'])
        img = '../nabirds/nabirds/images/' + z['image_name']
    #            print(img)
    #            plt.imshow(img)
        image = cv2.imread(img)
        
#        by0, by1, bx0, bx1, delta_h, delta_w, nbsl, image, rect = square_bbox(image, z)
        
                       
#        print(by0, by1, bx0, bx1)
        fig,ax = plt.subplots(1, figsize=(12,16))
        #image = plt.imread(img)
#        plt.imshow(cv2.cvtColor(image[by0:by1,bx0:bx1], cv2.COLOR_BGR2RGB))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        rect = patches.Rectangle((z['bb_x'],z['bb_y']),z['bb_width'],z['bb_height'],linewidth=3,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
        #rect2 = patches.Rectangle((bx0,by0),nbsl,nbsl,linewidth=1,edgecolor='b',facecolor='none')
        #ax.add_patch(rect2)
        plt.savefig('GHO_box.png')
        plt.show()
        
        
        
#%%
 for i in np.arange(402,6000,2000):
        print(i)
        z = full_frame.iloc[i]
        print(z['class_name_sp'])
        
        img = '../../images/' + z['image_name']
    #            print(img)
    #            plt.imshow(img)
        image = cv2.imread(img)
        
#        by0, by1, bx0, bx1, delta_h, delta_w, nbsl, image, rect = square_bbox(image, z)
        
                       
#        print(by0, by1, bx0, bx1)
        fig,ax = plt.subplots(1, figsize=(20,20))
        #image = plt.imread(img)
#        plt.imshow(cv2.cvtColor(image[by0:by1,bx0:bx1], cv2.COLOR_BGR2RGB))
        cropped = image[z['bb_y']:(z['bb_y']+z['bb_height']), z['bb_x']:(z['bb_x']+z['bb_width'])]

        plt.imshow(cv2.cvtColor(paint_to_square(cropped,z), cv2.COLOR_BGR2RGB))
#        rect = patches.Rectangle((z['bb_x'],z['bb_y']),z['bb_width'],z['bb_height'],linewidth=3,edgecolor='r',facecolor='none')
#        ax.add_patch(rect)
        #rect2 = patches.Rectangle((bx0,by0),nbsl,nbsl,linewidth=1,edgecolor='b',facecolor='none')
        #ax.add_patch(rect2)
        
        plt.show()
    
#%% Make hierarchy a dict
parents = {}
for idx in np.arange(len(hierarchy)):
    h = hierarchy.iloc[idx]
    child_id, parent_id = h['child_class_id'], h['parent_class_id']
    parents[child_id] = parent_id
           
#print(parents)
    
corresponding_class = {}
corresponding_class_name = {}
four_shell = 0
five_shell = 0
for idx in np.arange(len(hierarchy)):
    id_path = []
    label_path = []
    label_sub_path = []
    h = hierarchy.iloc[idx]
    current_class_id = h['child_class_id']
    belong_to_parent = 1
    while current_class_id in parents:
        id_path.append(current_class_id)
        label_path.append(classes.iloc[current_class_id]['class_name'])
        if ('(' not in classes.iloc[current_class_id]['class_name']) & (belong_to_parent == 1):
            corresponding_class[id_path[0]] = current_class_id
            corresponding_class_name[id_path[0]] = classes.iloc[current_class_id]['class_name']
            belong_to_parent = 0
        current_class_id = parents[current_class_id]
    print(idx, label_path)
    #print(corresponding_class_name)
        
    #if len(id_path) == 4:
    #    print(id_path, len(id_path), label_path)
    #    corresponding_class[id_path[0]] = id_path[1]
    #else:
    #    corresponding_class[id_path[0]] = id_path[0]      
#print(corresponding_class)

full_frame['class_id_sp'] = full_frame.apply(lambda row: corresponding_class[row.class_id], axis=1)
full_frame['class_name_sp'] = full_frame.apply(lambda row: corresponding_class_name[row.class_id], axis=1)
#full_frame['image_name_sp'] = full_frame['class_id_sp'] + full_frame.apply(lambda row: row.image_name.split('/')[1], axis=1)
full_frame['image_name_sp'] = full_frame.apply(lambda row: str('{:04d}'.format(row.class_id_sp)) + '/' + row.image_name.split('/')[1], axis=1)
full_frame['image_name_fname_only'] = full_frame.apply(lambda row: row.image_name.split('/')[1], axis=1)
#print(full_frame)

#%%
full_frame.to_csv('full_frame.csv')

#%%
Bird_list=full_frame.class_name_sp.unique()
print(Bird_list)
print(len(Bird_list))

#%%
max_variety = 0
target_photo = {}
for i in pd.unique(full_frame['class_id_sp']):
    z = full_frame.loc[full_frame['class_id_sp'].isin([i])]
#        max_variety = len(pd.unique(z['class_id']))
#        print(max_variety)
#        print(z.iloc[0])
    if len(pd.unique(z['class_id'])) == 1:
        for y in pd.unique(z['class_id']):
            target_photo[y] = 4
    if len(pd.unique(z['class_id'])) == 2:
        for y in pd.unique(z['class_id']):
            target_photo[y] = 2
    if len(pd.unique(z['class_id'])) == 3:
        for y in pd.unique(z['class_id']):
            target_photo[y] = 1
    if len(pd.unique(z['class_id'])) >= 4:
        for y in pd.unique(z['class_id']):
            target_photo[y] = 1

        
print(max_variety)

#%% Prepare images for website
web_dir = 'web_images/'
num_dir = len(os.listdir(web_dir))
os.mkdir(web_dir+str(num_dir))
has_photo = {}

# Copy accepted files
if num_dir > 1:
    for i in os.listdir(web_dir + str(num_dir-1)):
#        print(i)
        try:
            copyfile(web_dir+str(num_dir-1)+'/'+i, web_dir+'bird_img/'+i)
        except:
            pass

for i in pd.unique(full_frame['class_id']):
    has_photo[i] = 0
    
# Check if this species has photo
filelist = os.listdir(web_dir + 'bird_img')
z = full_frame.loc[full_frame['image_name_fname_only'].isin(filelist)]
#print(z)
for jj in pd.unique(z['class_id']):
    has_photo[jj] = len(z[z['class_id']==jj])

copy_counter = 0
for i in pd.unique(full_frame['class_id']):
#    print(i)
    if has_photo[i] < target_photo[i]:
        z = full_frame.loc[full_frame['class_id'].isin([i])].sample(n=(target_photo[i]-has_photo[i]))
        for idx, y in z.iterrows():
#            fname = str(full_frame.loc[full_frame['class_id'].isin([i])]['image_name'].values[0])
            fname= y['image_name']
            fname_d = fname.split('/')[-1]
    #    print(fname_d)
    #    print(full_frame.loc[full_frame['class_id'].isin([i])].sample(n=1)['image_name'].values[0])    
    #    print('../../images/'+fname)
            copyfile('../nabirds/nabirds/images/'+fname, web_dir+str(num_dir)+'/'+fname_d)
            copy_counter += 1
    
print(copy_counter)    
if copy_counter == 0:

    print("Photo selection completed!")

#%%
filelist = os.listdir(web_dir + 'bird_img')
z = full_frame.loc[full_frame['image_name_fname_only'].isin(filelist)]
z.to_csv('bird_img.csv')


#%%
b1='Snowy_Owl'
BI1 = 'static/bird_img/' + str(z.loc[z['class_name_sp'].isin([b1])].sample(n=1)['image_name_fname_only'].values[0])
print(BI1)
#%%
# Create 'train', 'val', 'test' folders
# Run train_val_test_split for each species
# write cropped images into respective folders with the same file name.

train_counter = 0
val_counter = 0
test_counter = 0
train_list = []
val_list = []
test_list = []
for idx, species in enumerate(Bird_list):
    df = full_frame[full_frame['class_name_sp'].isin([species])]
    df_length = len(df)
    train_length = round(df_length*3/4)
    val_length = (df_length - train_length) // 2
    test_length = df_length - train_length - val_length
    train_frame_this = df.iloc[:train_length]
    val_frame_this = df.iloc[train_length:train_length+val_length]
    test_frame_this = df.iloc[train_length+val_length:]
    train_counter += train_length
    val_counter += val_length
    test_counter += test_length
    #print(type(df.iloc[train_length+val_length:]))
    #print(train_length, val_length, test_length)
    train_list.append(train_frame_this)
    val_list.append(val_frame_this)
    test_list.append(test_frame_this)
    
train_frame = pd.concat(train_list, axis = 0)
val_frame = pd.concat(val_list, axis = 0)
test_frame = pd.concat(test_list, axis = 0)
print(train_counter, val_counter, test_counter)
print(len(train_frame), len(val_frame), len(test_frame))

#train_frame.to_csv('train.csv')
#val_frame.to_csv('val.csv')
#test_frame.to_csv('test.csv')

#y, X = 
#make_npy_many(full_frame, Bird_list, '../../data/y_555.npy', '../../data/X_555.npy', '../../data/L_555.npy')
#make_npy(df, 'Black_Throated_Sparrow', 'Harris_Sparrow', )





#%%
for idx in np.arange(len(val_frame)):
#for idx in np.arange(10):
    z = val_frame.iloc[idx]
    #if idx < 10:
    image = cv2.imread('../../images/' + z['image_name'])
#    by0, by1, bx0, bx1, delta_h, delta_w, nbsl, image, rect = square_bbox(image, z, verbose=0)
    cropped = image[z['bb_y']:(z['bb_y']+z['bb_height']), z['bb_x']:(z['bb_x']+z['bb_width'])]
    cv2.imwrite('../../data/val/'+z['image_name_sp'],paint_to_square(cropped, z))
#    cv2.imwrite('../../data/val/'+z['image_name_sp'],paint_to_square(image[by0:by1,bx0:bx1]))
    #new_x[i,:,:,:] = paint_to_square(cropped, z).astype(np.uint8)
    if idx % 100 == 0:
        print("Processing image {}.".format(idx))
#%%
for idx in np.arange(len(test_frame)):
    z = test_frame.iloc[idx]
    #if idx < 10:
    image = cv2.imread('../../images/' + z['image_name'])
#    by0, by1, bx0, bx1, delta_h, delta_w, nbsl, image, rect = square_bbox(image, z)
    cropped = image[z['bb_y']:(z['bb_y']+z['bb_height']), z['bb_x']:(z['bb_x']+z['bb_width'])]
    cv2.imwrite('../../data/test/'+z['image_name_sp'],paint_to_square(cropped, z))
#    cv2.imwrite('../../data/test/'+z['image_name_sp'],paint_to_square(image[by0:by1,bx0:bx1]))
    if idx % 100 == 0:
        print("Processing image {}.".format(idx))
#%%
for idx in np.arange(len(train_frame)):
    z = train_frame.iloc[idx]
    #if idx < 10:
    image = cv2.imread('../../images/' + z['image_name'])
#    by0, by1, bx0, bx1, delta_h, delta_w, nbsl, image, rect = square_bbox(image, z)
    cropped = image[z['bb_y']:(z['bb_y']+z['bb_height']), z['bb_x']:(z['bb_x']+z['bb_width'])]
    cv2.imwrite('../../data/train/'+z['image_name_sp'],paint_to_square(cropped, z))
#    cv2.imwrite('../../data/train/'+z['image_name_sp'],paint_to_square(image[by0:by1,bx0:bx1]))
    if idx % 100 == 0:
        print("Processing image {}.".format(idx))        