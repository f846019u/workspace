import caffe
from caffe.proto import caffe_pb2
from caffe.io import blobproto_to_array
import numpy as np # for array
import glob
import os.path

# directory = "C:/dev/caffe/caffe-windows"
net_path = "./examples/mytest/lenet/deploy.prototxt"
model_path = "./examples/mytest/mytest_iter_100000.caffemodel"
mean_path = "./data/mytest/mean.binaryproto"
eval_path = "./data/mytest/eval.txt"

# -- 
#net_path : deploy.prototxt, model_path : caffemodel. 識別なので、Caffe.TESTと記載
net = caffe.Net(net_path, model_path,caffe.TEST)
# --

# for preprocess
transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
# channel_swapはRGBをσ(RGB) = BGRにしたいときに用いる。
# 対して、set_transposeは、[H, W, K] => [K, H, W]としたいときとかに用いる。
transformer.set_transpose('data', (2,0,1)) 

mean_blob = caffe_pb2.BlobProto()
with open(mean_path, "rb") as f:
    mean_blob.ParseFromString(f.read())
mean_array = blobproto_to_array(mean_blob)
mean_array = np.asarray(mean_blob.data).reshape((mean_blob.channels, mean_blob.height, mean_blob.width))
transformer.set_mean('data', mean_array)

#上記の説明参照
transformer.set_raw_scale('data', 255)
# scaling to 1/255 = 0.003..
transformer.set_input_scale('data', 0.00390625)

# sample test for 11181 images
cnt = 0 #number of the collect answer
num = 11181

numbers = [str(i+60000).zfill(7) for i in range(1,num+1)]
filename = [s + ".bmp" for s in numbers]
images = "data/mytest/check"


import csv
reader = csv.reader(open(eval_path), lineterminator = ' ')
ansList = []
for row in reader:
    i = 0
    (fl, i) = row[0].split(' ')
    ansList.append((int(i)))


for i, fname in enumerate(filename):

    fn = images + "/" + fname

    image = caffe.io.load_image(fn, color = False)
    # image : (H×W×K) ndarray  => set_transposeが効いて、outputは(K×W×H) ndarray型になる。
    proc = transformer.preprocess('data', image)
   # (K×W×H) ndarray のinputデータ : proc
   # net.inputs[0]はdeploy.prototxtのdataレイヤーに記載の'data'(topプロパティ)
    out = net.forward_all(**{net.inputs[0]:proc})
    # net.outputs[0]はdeploy.prototxtのdataレイヤーに記載の'prob'(topプロパティ)
    predictions = out[net.outputs[0]]
    answer = np.argmax(predictions)

    print predictions,  "<= prob"
    print net.blobs['ip2'].data ," <= ip2"

    print fn, i,", true answer =>", ansList[i],", eval =>", answer, ansList[i] == answer
    cnt = cnt + 1 if (ansList[i] == answer) else cnt
print cnt, "/", num
