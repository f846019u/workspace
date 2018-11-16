#!/usr/bin/env python

import caffe

caffe.set_mode_cpu()

solver_prototxt = '../china_lenet_solver.prototxt'
solver = caffe.SGDsolver(solver_prototxt)

pretrained_model = '../weights/china_lenet/china_lenet_value0_iter_20000.caffemodel'
solver.net.copy_from(pretrained_model)

from caffe.proto import caffe_pb2
import google.protobuf as pb2
from google.protobuf import text_format

#solverのパラメータを読み込む
solver_param = caffe_pb2.SolverParameter()
with open(solver_prototxt, 'rt') as f:
    pb2.text_format.Merge(f.read(), solver_param)


max_iters = solver_param.max_iter

while solver.iter < max_iters:
    solver.step(1)

filename = solver_param.snapshot_prefix + '_iter_{:d}'.format(solver.iter) + '.caffemodel'
#net.save(filename)

print(filename)

