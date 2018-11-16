#!/bin/sh

echo 'check.pyの実行'

ipython2 ./value1/check_china_lenet.py > ./log/china_lenet_value1_miss_num.txt

ipython2 ./value1/check_china_lenet_gap.py > ./log/china_lenet_gap_value1_miss_num.txt

ipython2 ./value1/check_china_lenet_dp2.py > ./log/china_lenet_dp2_value1_miss_num.txt

ipython2 ./value1/check_china_lenet_dp2_gap.py > ./log/china_lenet_dp2_gap_value1_miss_num.txt

ipython2 ./value1/check_china_lenet_dp23.py > ./log/china_lenet_dp23_value1_miss_num.txt

ipython2 ./value1/check_china_lenet_dp234.py > ./log/china_lenet_dp234_value1_miss_num.txt

echo 'check.py終了'
