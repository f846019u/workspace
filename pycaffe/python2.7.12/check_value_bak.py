#!/usr/bin/env python

import numpy as np
import sys
import os
import caffe
import cv2 as cv

#評価する番号
VALUE_NUM = '0'
MODEL_NAME = 'china_lenet'

MODEL_FILE = '../../' + MODEL_NAME  + '_deploy.prototxt'
PRETRAINED = '../../weights/' + MODEL_NAME + '/' + MODEL_NAME + '_value' + VALUE_NUM + '_iter_20000.caffemodel'
IMAGE_FILE = '/home/higaki/china_data/image_0/test/alp/A/0202_SN1_3.png'

if not os.path.isfile(MODEL_FILE):
    print("error: caffe model load...")

if not os.path.isfile(PRETRAINED):
    print("error: pre-trained caffe model...")

if not os.path.isfile(IMAGE_FILE):
    print("error: image_file not open.")


##テストデータのnumpyファイル作成##

#width, height, test_class定義
width = 14
height = 20
test_class = 36

#dir_listにPATHのフォルダを入れていく
PATH = '/home/higaki/china_data/image_' + VALUE_NUM + '/test/'
tmp = os.listdir(PATH)
tmp = sorted([x for x in tmp if os.path.isdir(PATH + x)])
dir_list = tmp 
print(dir_list)

X_test = []
Y_test = []
image_name = []
label = 0

# alp or num を処理
for alp_num in dir_list:
    if str(alp_num) == 'num':
        label = 0
    if str(alp_num) == 'alp':
        label = 10
    
    dir_list = os.listdir(PATH + str(alp_num) + '/')
    
    # A~Z or 0~9 を処理 
    for dir_name in dir_list :
        file_list = os.listdir(PATH + str(alp_num) + '/' + str(dir_name))
        #print file_list
        #print len(file_list)
        
        # それぞれの画像を処理
        for file_name in file_list:
            if file_name.endswith('.png'):
                #image = cv.imread(PATH + str(alp_num) + '/' + str(dir_name) + '/' + file_name, 0)
                #image = cv.resize(image, (width, height))
                #cv.normalize(image, image, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
                #image = image / 255.
                #X_test.append(image)
                Y_test.append(label)
                image_name.append(PATH + str(alp_num) + '/' + str(dir_name) + '/' + file_name)
            
        label = label + 1

#print Y_test

X_test = np.asarray(X_test)
Y_test = np.asarray(Y_test)

print(X_test.shape)
print(Y_test.shape)
print(len(image_name))
print image_name[0]
image_name.sort()
print image_name[0]


## テストデータのどこが間違えているかを確認 ##

caffe.set_mode_cpu()

net = caffe.Classifier(MODEL_FILE, PRETRAINED, image_dims=(20, 14))

for (Y, name) in zip(Y_test, image_name):
    #print name
    
    input_image = caffe.io.load_image(name, color=False)
    prediction = net.predict([input_image], False)

    #print("prediction shape: {}".format(prediction[0].shape))
    #print("predicted class: {}".format(prediction[0].argmax()) + ' ' + str(Y))

    


