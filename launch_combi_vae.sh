#!/usr/bin/env bash

## Experiments for single mode generalization -- size
#python vae2.py --gpu=0 --dataset=crand_30_0_3 --beta=$1 --run=$2 &
#python vae2.py --gpu=1 --dataset=crand_30_0_4 --beta=$1 --run=$2 &
#python vae2.py --gpu=2 --dataset=crand_30_0_5 --beta=$1 --run=$2 &
#python vae2.py --gpu=3 --dataset=crand_30_0_6 --beta=$1 --run=$2 &
#python vae2.py --gpu=4 --dataset=crand_30_0_7 --beta=$1 --run=$2 &
#
## Experiments for single mode generalization -- location
#python vae2.py --gpu=5 --dataset=crand_30_1_3 --beta=$1 --run=$2 &
#python vae2.py --gpu=6 --dataset=crand_30_1_4 --beta=$1 --run=$2 &
#python vae2.py --gpu=7 --dataset=crand_30_1_5 --beta=$1 --run=$2 &
#python vae2.py --gpu=0 --dataset=crand_30_1_6 --beta=$1 --run=$2 &
#python vae2.py --gpu=1 --dataset=crand_30_1_7 --beta=$1 --run=$2 &
#
## Experiments for single mode generalization -- color
#python vae2.py --gpu=2 --dataset=crand_30_3_1 --beta=$1 --run=$2 &
#python vae2.py --gpu=3 --dataset=crand_30_3_3 --beta=$1 --run=$2 &
#python vae2.py --gpu=4 --dataset=crand_30_3_4 --beta=$1 --run=$2 &
#python vae2.py --gpu=5 --dataset=crand_30_3_5 --beta=$1 --run=$2 &
#python vae2.py --gpu=6 --dataset=crand_30_3_6 --beta=$1 --run=$2 &
#python vae2.py --gpu=7 --dataset=crand_30_3_7 --beta=$1 --run=$2 &
#python vae2.py --gpu=0 --dataset=crand_30_3_8 --beta=$1 --run=$2 &
#python vae2.py --gpu=1 --dataset=crand_30_3_9 --beta=$1 --run=$2 &
#
## Experiments for prototypical enhancement
#python vae2.py --gpu=2 --dataset=crand_30_3_34 --beta=$1 --run=$2 &
#python vae2.py --gpu=3 --dataset=crand_30_3_35 --beta=$1 --run=$2 &
#python vae2.py --gpu=4 --dataset=crand_30_3_36 --beta=$1 --run=$2 &
#python vae2.py --gpu=5 --dataset=crand_30_3_37 --beta=$1 --run=$2 &
#python vae2.py --gpu=6 --dataset=crand_30_3_38 --beta=$1 --run=$2 &
#python vae2.py --gpu=7 --dataset=crand_30_3_39 --beta=$1 --run=$2 &

# Experiments for stability
python vae2.py --gpu=0 --dataset=crand_3_3_349 --beta=$1 --run=$2 &
python vae2.py --gpu=1 --dataset=crand_6_3_349 --beta=$1 --run=$2 &
python vae2.py --gpu=2 --dataset=crand_9_3_349 --beta=$1 --run=$2 &
python vae2.py --gpu=3 --dataset=crand_18_3_349 --beta=$1 --run=$2 &
python vae2.py --gpu=4 --dataset=crand_30_3_349 --beta=$1 --run=$2 &
python vae2.py --gpu=5 --dataset=crand_60_3_349 --beta=$1 --run=$2 &
python vae2.py --gpu=6 --dataset=crand_90_3_349 --beta=$1 --run=$2 &
python vae2.py --gpu=7 --dataset=crand_150_3_349 --beta=$1 --run=$2 &
#
python vae2.py --gpu=0 --dataset=crand_3_1_459 --beta=$1 --run=$2 &
python vae2.py --gpu=1 --dataset=crand_6_1_459 --beta=$1 --run=$2 &
python vae2.py --gpu=2 --dataset=crand_9_1_459 --beta=$1 --run=$2 &
python vae2.py --gpu=3 --dataset=crand_18_1_459 --beta=$1 --run=$2 &
python vae2.py --gpu=4 --dataset=crand_30_1_459 --beta=$1 --run=$2 &
python vae2.py --gpu=5 --dataset=crand_60_1_459 --beta=$1 --run=$2 &
python vae2.py --gpu=6 --dataset=crand_90_1_459 --beta=$1 --run=$2 &
python vae2.py --gpu=7 --dataset=crand_150_1_459 --beta=$1 --run=$2 &

python vae2.py --gpu=0 --dataset=crand_3_0_349 --beta=$1 --run=$2 &
python vae2.py --gpu=1 --dataset=crand_6_0_349 --beta=$1 --run=$2 &
python vae2.py --gpu=2 --dataset=crand_9_0_349 --beta=$1 --run=$2 &
python vae2.py --gpu=3 --dataset=crand_18_0_349 --beta=$1 --run=$2 &
python vae2.py --gpu=4 --dataset=crand_30_0_349 --beta=$1 --run=$2 &
python vae2.py --gpu=5 --dataset=crand_60_0_349 --beta=$1 --run=$2 &
python vae2.py --gpu=6 --dataset=crand_90_0_349 --beta=$1 --run=$2 &
python vae2.py --gpu=7 --dataset=crand_150_0_349 --beta=$1 --run=$2 &

## Experiments for random memorization
#python vae2.py --gpu=0 --dataset=crand_10 --beta=$1 --run=3 &
#python vae2.py --gpu=1 --dataset=crand_15 --beta=$1 --run=3 &
#python vae2.py --gpu=2 --dataset=crand_20 --beta=$1 --run=3 &
#python vae2.py --gpu=3 --dataset=crand_30 --beta=$1 --run=3 &
#python vae2.py --gpu=4 --dataset=crand_50 --beta=$1 --run=3 &
#python vae2.py --gpu=5 --dataset=crand_60 --beta=$1 --run=3 &
#python vae2.py --gpu=6 --dataset=crand_75 --beta=$1 --run=3 &
#python vae2.py --gpu=7 --dataset=crand_100 --beta=$1 --run=3 &
#python vae2.py --gpu=0 --dataset=crand_150 --beta=$1 --run=3 &
#python vae2.py --gpu=1 --dataset=crand_200 --beta=$1 --run=3 &
#python vae2.py --gpu=2 --dataset=crand_300 --beta=$1 --run=3 &
#python vae2.py --gpu=3 --dataset=crand_400 --beta=$1 --run=3 &
#
#python vae2.py --gpu=0 --dataset=crand_10 --beta=$1 --run=4 &
#python vae2.py --gpu=1 --dataset=crand_15 --beta=$1 --run=4 &
#python vae2.py --gpu=2 --dataset=crand_20 --beta=$1 --run=4 &
#python vae2.py --gpu=3 --dataset=crand_30 --beta=$1 --run=4 &
#python vae2.py --gpu=4 --dataset=crand_50 --beta=$1 --run=4 &
#python vae2.py --gpu=5 --dataset=crand_60 --beta=$1 --run=4 &
#python vae2.py --gpu=6 --dataset=crand_75 --beta=$1 --run=4 &
#python vae2.py --gpu=7 --dataset=crand_100 --beta=$1 --run=4 &
#python vae2.py --gpu=4 --dataset=crand_150 --beta=$1 --run=4 &
#python vae2.py --gpu=5 --dataset=crand_200 --beta=$1 --run=4 &
#python vae2.py --gpu=6 --dataset=crand_300 --beta=$1 --run=4 &
#python vae2.py --gpu=7 --dataset=crand_400 --beta=$1 --run=4 &