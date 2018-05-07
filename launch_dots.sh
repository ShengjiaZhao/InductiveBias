#!/usr/bin/env bash

#python gan.py --gpu=0 --dataset=count_1 &
#python gan.py --gpu=1 --dataset=count_2 &
#python gan.py --gpu=2 --dataset=count_3 &
#python gan.py --gpu=3 --dataset=count_1_2 &
#python gan.py --gpu=4 --dataset=count_2_3 &
#python gan.py --gpu=5 --dataset=count_1_3 &
#python gan.py --gpu=6 --dataset=count_1_2_3 &
#python gan.py --gpu=7 --dataset=count_4 &


python gan.py --gpu=0 --dataset=disjoint_count_1 &
python gan.py --gpu=1 --dataset=disjoint_count_2 &
python gan.py --gpu=2 --dataset=disjoint_count_3 &
python gan.py --gpu=3 --dataset=disjoint_count_4 &
python gan.py --gpu=4 --dataset=disjoint_count_5 &
python gan.py --gpu=5 --dataset=disjoint_count_6 &
python gan.py --gpu=6 --dataset=disjoint_count_7 &
python gan.py --gpu=7 --dataset=disjoint_count_8 &
python gan.py --gpu=0 --dataset=disjoint_count_9 &
python gan.py --gpu=1 --dataset=disjoint_count_10 &
python gan.py --gpu=2 --dataset=disjoint_count_11 &
python gan.py --gpu=3 --dataset=disjoint_count_12 &

python vae.py --gpu=0 --z_dim=4 --dataset=disjoint_count_1 &
python vae.py --gpu=1 --z_dim=6 --dataset=disjoint_count_2 &
python vae.py --gpu=2 --z_dim=8 --dataset=disjoint_count_3 &
python vae.py --gpu=3 --z_dim=10 --dataset=disjoint_count_4 &
python vae.py --gpu=4 --z_dim=12 --dataset=disjoint_count_5 &
python vae.py --gpu=5 --z_dim=14 --dataset=disjoint_count_6 &
python vae.py --gpu=6 --z_dim=16 --dataset=disjoint_count_7 &
python vae.py --gpu=7 --z_dim=18 --dataset=disjoint_count_8 &
python vae.py --gpu=4 --z_dim=20 --dataset=disjoint_count_9 &
python vae.py --gpu=5 --z_dim=22 --dataset=disjoint_count_10 &
python vae.py --gpu=6 --z_dim=24 --dataset=disjoint_count_11 &
python vae.py --gpu=7 --z_dim=26 --dataset=disjoint_count_12 &