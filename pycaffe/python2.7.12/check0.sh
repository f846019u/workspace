#!/bin/sh

echo 'check.pyの実行'

ipython2 ./value0/check_china_lenet.py > ./log/china_lenet_value0_miss_num.txt

ipython2 ./value0/check_china_lenet_gap.py > ./log/china_lenet_gap_value0_miss_num.txt

ipython2 ./value0/check_china_lenet_dp2.py > ./log/china_lenet_dp2_value0_miss_num.txt

ipython2 ./value0/check_china_lenet_dp2_gap.py > ./log/china_lenet_dp2_gap_value0_miss_num.txt

ipython2 ./value0/check_china_lenet_dp23.py > ./log/china_lenet_dp23_value0_miss_num.txt

ipython2 ./value0/check_china_lenet_dp234.py > ./log/china_lenet_dp234_value0_miss_num.txt

echo 'check.py終了'
