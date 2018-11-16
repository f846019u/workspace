#!/usr/bin/env python

import numpy as np
import sys
import os
import caffe

MODEL_FILE = '../../china_lenet_deploy.prototxt'
PRETRAINED = '../../weights/china_lenet/china_lenet_value0_iter_20000.caffemodel'
IMAGE_FILE = '/home/higaki/china_data/image_0/test/alp/A/0202_SN1_3.png'

print(MODEL_FILE)
print(PRETRAINED)
print(IMAGE_FILE)

if not os.path.isfile(MODEL_FILE):
    print("error: caffe model load...")

if not os.path.isfile(PRETRAINED):
    print("error: pre-trained caffe model...")


if not os.path.isfile(IMAGE_FILE):
    print("error: image_file not open.")

caffe.set_mode_cpu()

net = caffe.Classifier(MODEL_FILE, PRETRAINED, image_dims=(20, 14))


input_image = caffe.io.load_image(IMAGE_FILE, color=False)

prediction = net.predict([input_image], False)

print("prediction shape: {}".format(prediction[0].shape))
print("predicted class: {}".format(prediction[0].argmax()))



