#!/usr/bin/env python

import numpy as np
import sys
import os
import caffe

MODEL_FILE = '../china_lenet_deploy.prototxt'
PRETRAINED = '../weights/china_lenet/china_lenet_value0_iter_20000.caffemodel'

print(MODEL_FILE)
print(PRETRAINED)

argvs = sys.argv
argc = len(argvs)

if not os.path.isfile(PRETRAINED):
    print("error: pre-trained caffe model...")

IMAGE_FILE = argvs[1]

caffe.set_mode_cpu()

net = caffe.Classifier(MODEL_FILE, 
                       PRETRAINED, 
                       channel_swap=[0], 
                       image_dims=(20, 14))

#input_image = caffe.io.load_image(IMAGE_FILE, color=False) 

#prediction = net.predict([input_image], False)
