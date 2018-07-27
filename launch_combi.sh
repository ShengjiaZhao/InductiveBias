#!/usr/bin/env bash

# Experiments for single mode generalization -- size
#python gan2.py --gpu=0 --dataset=crand_30_0_3 --model=$1 --run=$2 &
#python gan2.py --gpu=1 --dataset=crand_30_0_4 --model=$1 --run=$2 &
#python gan2.py --gpu=2 --dataset=crand_30_0_5 --model=$1 --run=$2 &
#python gan2.py --gpu=3 --dataset=crand_30_0_6 --model=$1 --run=$2 &
#python gan2.py --gpu=4 --dataset=crand_30_0_7 --model=$1 --run=$2 &
#
## Experiments for single mode generalization -- location
#python gan2.py --gpu=5 --dataset=crand_30_1_3 --model=$1 --run=$2 &
#python gan2.py --gpu=6 --dataset=crand_30_1_4 --model=$1 --run=$2 &
#python gan2.py --gpu=7 --dataset=crand_30_1_5 --model=$1 --run=$2 &
#python gan2.py --gpu=0 --dataset=crand_30_1_6 --model=$1 --run=$2 &
#python gan2.py --gpu=1 --dataset=crand_30_1_7 --model=$1 --run=$2 &
#
## Experiments for single mode generalization -- color
#python gan2.py --gpu=2 --dataset=crand_30_3_1 --model=$1 --run=$2 &
#python gan2.py --gpu=3 --dataset=crand_30_3_3 --model=$1 --run=$2 &
#python gan2.py --gpu=4 --dataset=crand_30_3_4 --model=$1 --run=$2 &
#python gan2.py --gpu=5 --dataset=crand_30_3_5 --model=$1 --run=$2 &
#python gan2.py --gpu=6 --dataset=crand_30_3_6 --model=$1 --run=$2 &
#python gan2.py --gpu=7 --dataset=crand_30_3_7 --model=$1 --run=$2 &
#python gan2.py --gpu=0 --dataset=crand_30_3_8 --model=$1 --run=$2 &
#python gan2.py --gpu=1 --dataset=crand_30_3_9 --model=$1 --run=$2 &
#
## Experiments for prototypical enhancement
#python gan2.py --gpu=2 --dataset=crand_30_3_34 --model=$1 --run=$2 &
#python gan2.py --gpu=3 --dataset=crand_30_3_35 --model=$1 --run=$2 &
#python gan2.py --gpu=4 --dataset=crand_30_3_36 --model=$1 --run=$2 &
#python gan2.py --gpu=5 --dataset=crand_30_3_37 --model=$1 --run=$2 &
#python gan2.py --gpu=6 --dataset=crand_30_3_38 --model=$1 --run=$2 &
#python gan2.py --gpu=7 --dataset=crand_30_3_39 --model=$1 --run=$2 &

## Experiments for stability
python gan2.py --gpu=0 --dataset=crand_3_3_349 --model=$1 --run=$2 &
python gan2.py --gpu=1 --dataset=crand_6_3_349 --model=$1 --run=$2 &
python gan2.py --gpu=2 --dataset=crand_9_3_349 --model=$1 --run=$2 &
python gan2.py --gpu=3 --dataset=crand_18_3_349 --model=$1 --run=$2 &
python gan2.py --gpu=4 --dataset=crand_30_3_349 --model=$1 --run=$2 &
python gan2.py --gpu=5 --dataset=crand_60_3_349 --model=$1 --run=$2 &
python gan2.py --gpu=6 --dataset=crand_90_3_349 --model=$1 --run=$2 &
python gan2.py --gpu=7 --dataset=crand_150_3_349 --model=$1 --run=$2 &
#
python gan2.py --gpu=0 --dataset=crand_3_1_459 --model=$1 --run=$2 &
python gan2.py --gpu=1 --dataset=crand_6_1_459 --model=$1 --run=$2 &
python gan2.py --gpu=2 --dataset=crand_9_1_459 --model=$1 --run=$2 &
python gan2.py --gpu=3 --dataset=crand_18_1_459 --model=$1 --run=$2 &
python gan2.py --gpu=4 --dataset=crand_30_1_459 --model=$1 --run=$2 &
python gan2.py --gpu=5 --dataset=crand_60_1_459 --model=$1 --run=$2 &
python gan2.py --gpu=6 --dataset=crand_90_1_459 --model=$1 --run=$2 &
python gan2.py --gpu=7 --dataset=crand_150_1_459 --model=$1 --run=$2 &

python gan2.py --gpu=0 --dataset=crand_3_0_349 --model=$1 --run=$2 &
python gan2.py --gpu=1 --dataset=crand_6_0_349 --model=$1 --run=$2 &
python gan2.py --gpu=2 --dataset=crand_9_0_349 --model=$1 --run=$2 &
python gan2.py --gpu=3 --dataset=crand_18_0_349 --model=$1 --run=$2 &
python gan2.py --gpu=4 --dataset=crand_30_0_349 --model=$1 --run=$2 &
python gan2.py --gpu=5 --dataset=crand_60_0_349 --model=$1 --run=$2 &
python gan2.py --gpu=6 --dataset=crand_90_0_349 --model=$1 --run=$2 &
python gan2.py --gpu=7 --dataset=crand_150_0_349 --model=$1 --run=$2 &

## Experiments for random memorization
#python gan2.py --gpu=0 --dataset=crand_10 --model=$1 --run=4 &
#python gan2.py --gpu=1 --dataset=crand_15 --model=$1 --run=4 &
#python gan2.py --gpu=2 --dataset=crand_20 --model=$1 --run=4 &
#python gan2.py --gpu=3 --dataset=crand_30 --model=$1 --run=4 &
#python gan2.py --gpu=4 --dataset=crand_50 --model=$1 --run=4 &
#python gan2.py --gpu=5 --dataset=crand_60 --model=$1 --run=4 &
#python gan2.py --gpu=6 --dataset=crand_75 --model=$1 --run=4 &
#python gan2.py --gpu=7 --dataset=crand_100 --model=$1 --run=4 &
#python gan2.py --gpu=0 --dataset=crand_150 --model=$1 --run=4 &
#python gan2.py --gpu=1 --dataset=crand_200 --model=$1 --run=4 &
#python gan2.py --gpu=2 --dataset=crand_300 --model=$1 --run=4 &
#python gan2.py --gpu=3 --dataset=crand_400 --model=$1 --run=4 &
#
#python gan2.py --gpu=0 --dataset=crand_10 --model=$1 --run=3 &
#python gan2.py --gpu=1 --dataset=crand_15 --model=$1 --run=3 &
#python gan2.py --gpu=2 --dataset=crand_20 --model=$1 --run=3 &
#python gan2.py --gpu=3 --dataset=crand_30 --model=$1 --run=3 &
#python gan2.py --gpu=4 --dataset=crand_50 --model=$1 --run=3 &
#python gan2.py --gpu=5 --dataset=crand_60 --model=$1 --run=3 &
#python gan2.py --gpu=6 --dataset=crand_75 --model=$1 --run=3 &
#python gan2.py --gpu=7 --dataset=crand_100 --model=$1 --run=3 &
#python gan2.py --gpu=4 --dataset=crand_150 --model=$1 --run=3 &
#python gan2.py --gpu=5 --dataset=crand_200 --model=$1 --run=3 &
#python gan2.py --gpu=6 --dataset=crand_300 --model=$1 --run=3 &
#python gan2.py --gpu=7 --dataset=crand_400 --model=$1 --run=3 &