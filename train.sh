#!/usr/bash


~/caffe/build/tools/caffe train -solver china_lenet_solver.prototxt

~/caffe/build/tools/caffe train -solver china_lenet_gap_solver.prototxt

~/caffe/build/tools/caffe train -solver china_lenet_dp2_solver.prototxt

~/caffe/build/tools/caffe train -solver china_lenet_dp2_gap_solver.prototxt

exit 0
