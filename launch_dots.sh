#!/usr/bin/env bash

#python gan.py --gpu=0 --dataset=count_1 &
#python gan.py --gpu=1 --dataset=count_2 &
#python gan.py --gpu=2 --dataset=count_3 &
#python gan.py --gpu=3 --dataset=count_1_2 &
#python gan.py --gpu=4 --dataset=count_2_3 &
#python gan.py --gpu=5 --dataset=count_1_3 &
#python gan.py --gpu=6 --dataset=count_1_2_3 &
#python gan.py --gpu=7 --dataset=count_4 &


#python gan.py --gpu=0 --dataset=disjoint_count_1 --model=gaussian_wgan &
#python gan.py --gpu=1 --dataset=disjoint_count_2 --model=gaussian_wgan  &
#python gan.py --gpu=2 --dataset=disjoint_count_3 --model=gaussian_wgan &
#python gan.py --gpu=3 --dataset=disjoint_count_4 --model=gaussian_wgan  &
#python gan.py --gpu=4 --dataset=disjoint_count_5 --model=gaussian_wgan  &
#python gan.py --gpu=5 --dataset=disjoint_count_6 --model=gaussian_wgan  &
#python gan.py --gpu=6 --dataset=disjoint_count_7 --model=gaussian_wgan  &
#python gan.py --gpu=7 --dataset=disjoint_count_8 --model=gaussian_wgan  &
#python gan.py --gpu=0 --dataset=disjoint_count_9 --model=gaussian_wgan  &
#python gan.py --gpu=1 --dataset=disjoint_count_10 --model=gaussian_wgan  &
#python gan.py --gpu=2 --dataset=disjoint_count_11 --model=gaussian_wgan  &
#python gan.py --gpu=3 --dataset=disjoint_count_12 --model=gaussian_wgan  &

#python gan.py --gpu=0 --dataset=disjoint_count_3_10 --model=gaussian_wgan  &
#python gan.py --gpu=1 --dataset=disjoint_count_4_9 --model=gaussian_wgan  &
#python gan.py --gpu=2 --dataset=disjoint_count_5_8 --model=gaussian_wgan  &
#python gan.py --gpu=3 --dataset=disjoint_count_6_7 --model=gaussian_wgan  &
#python gan.py --gpu=4 --dataset=disjoint_count_3_10 --model=gaussian_dcgan  &
#python gan.py --gpu=5 --dataset=disjoint_count_4_9 --model=gaussian_dcgan  &
#python gan.py --gpu=6 --dataset=disjoint_count_5_8 --model=gaussian_dcgan  &
#python gan.py --gpu=7 --dataset=disjoint_count_6_7 --model=gaussian_dcgan  &

#python gan.py --gpu=0 --dataset=disjoint_count_3_6_9 --model=gaussian_wgan  &
#python gan.py --gpu=1 --dataset=disjoint_count_4_6_8 --model=gaussian_wgan  &
#python gan.py --gpu=2 --dataset=disjoint_count_3_6_9_10 --model=gaussian_wgan  &
#python gan.py --gpu=3 --dataset=disjoint_count_3_4_5_6_7_8 --model=gaussian_wgan  &
#
#python gan.py --gpu=4 --dataset=disjoint_count_3_6_9 --model=gaussian_dcgan  &
#python gan.py --gpu=5 --dataset=disjoint_count_4_6_8 --model=gaussian_dcgan  &
#python gan.py --gpu=6 --dataset=disjoint_count_3_6_9_10 --model=gaussian_dcgan  &
#python gan.py --gpu=7 --dataset=disjoint_count_3_4_5_6_7_8 --model=gaussian_dcgan  &

#python gan.py --gpu=4 --dataset=disjoint_count_1 --model=gaussian_dcgan &
#python gan.py --gpu=5 --dataset=disjoint_count_2 --model=gaussian_dcgan  &
#python gan.py --gpu=6 --dataset=disjoint_count_3 --model=gaussian_dcgan &
#python gan.py --gpu=7 --dataset=disjoint_count_4 --model=gaussian_dcgan  &
#python gan.py --gpu=4 --dataset=disjoint_count_5 --model=gaussian_dcgan  &
#python gan.py --gpu=5 --dataset=disjoint_count_6 --model=gaussian_dcgan  &
#python gan.py --gpu=6 --dataset=disjoint_count_7 --model=gaussian_dcgan  &
#python gan.py --gpu=7 --dataset=disjoint_count_8 --model=gaussian_dcgan  &
#python gan.py --gpu=0 --dataset=disjoint_count_9 --model=gaussian_dcgan  &
#python gan.py --gpu=1 --dataset=disjoint_count_10 --model=gaussian_dcgan  &
#python gan.py --gpu=2 --dataset=disjoint_count_11 --model=gaussian_dcgan  &
#python gan.py --gpu=3 --dataset=disjoint_count_12 --model=gaussian_dcgan  &

#python gan.py --gpu=0 --dataset=color_gap_2 --model=gaussian_wgan &
#python gan.py --gpu=1 --dataset=color_gap_5 --model=gaussian_wgan &
#python gan.py --gpu=2 --dataset=color_gap_10 --model=gaussian_wgan &
#python gan.py --gpu=3 --dataset=color_gap_20 --model=gaussian_wgan &
#python gan.py --gpu=0 --dataset=color_gap_40 --model=gaussian_wgan &
#python gan.py --gpu=1 --dataset=color_gap_45 --model=gaussian_wgan &
#python gan.py --gpu=4 --dataset=color_gap_2 --model=gaussian_dcgan &
#python gan.py --gpu=5 --dataset=color_gap_5 --model=gaussian_dcgan &
#python gan.py --gpu=6 --dataset=color_gap_10 --model=gaussian_dcgan &
#python gan.py --gpu=7 --dataset=color_gap_20 --model=gaussian_dcgan &
#python gan.py --gpu=2 --dataset=color_gap_40 --model=gaussian_dcgan &
#python gan.py --gpu=3 --dataset=color_gap_45 --model=gaussian_dcgan &
#python vae.py --gpu=4 --dataset=color_gap_2 --beta=50.0 &
#python vae.py --gpu=5 --dataset=color_gap_5 --beta=50.0 &
#python vae.py --gpu=6 --dataset=color_gap_10 --beta=50.0 &
#python vae.py --gpu=7 --dataset=color_gap_20 --beta=50.0 &
#python vae.py --gpu=6 --dataset=color_gap_40 --beta=50.0 &
#python vae.py --gpu=7 --dataset=color_gap_45 --beta=50.0 &

#
python vae.py --gpu=0 --z_dim=10 --dataset=disjoint_count_1 --beta=$1 --run=$2 &
python vae.py --gpu=1 --z_dim=10 --dataset=disjoint_count_2 --beta=$1 --run=$2 &
python vae.py --gpu=2 --z_dim=10 --dataset=disjoint_count_3 --beta=$1 --run=$2 &
python vae.py --gpu=3 --z_dim=10 --dataset=disjoint_count_4 --beta=$1 --run=$2 &
python vae.py --gpu=4 --z_dim=12 --dataset=disjoint_count_5 --beta=$1 --run=$2 &
python vae.py --gpu=5 --z_dim=14 --dataset=disjoint_count_6 --beta=$1 --run=$2 &
python vae.py --gpu=6 --z_dim=16 --dataset=disjoint_count_7 --beta=$1 --run=$2 &
python vae.py --gpu=7 --z_dim=18 --dataset=disjoint_count_8 --beta=$1 --run=$2 &
python vae.py --gpu=4 --z_dim=20 --dataset=disjoint_count_9 --beta=$1 --run=$2 &
python vae.py --gpu=5 --z_dim=22 --dataset=disjoint_count_10 --beta=$1 --run=$2 &
python vae.py --gpu=6 --z_dim=24 --dataset=disjoint_count_11 --beta=$1 --run=$2 &
python vae.py --gpu=7 --z_dim=26 --dataset=disjoint_count_12 --beta=$1 --run=$2 &

#python gan.py --gpu=4 --dataset=size_0 &
#python gan.py --gpu=5 --dataset=size_4 &
#python gan.py --gpu=6 --dataset=size_8 &
#python gan.py --gpu=7 --dataset=size_12 &
#python vae.py --gpu=0 --dataset=size_0 --beta=$1 &
#python vae.py --gpu=1 --dataset=size_4 --beta=$1 &
#python vae.py --gpu=2 --dataset=size_8 --beta=$1 &
#python vae.py --gpu=3 --dataset=size_12 --beta=$1 &

#python gan.py --gpu=0 --dataset=color_count_10_1 &
#python gan.py --gpu=1 --dataset=color_count_10_3 &
#python gan.py --gpu=2 --dataset=color_count_10_5 &
#python gan.py --gpu=3 --dataset=color_count_10_7 &
#python gan.py --gpu=4 --dataset=color_count_10_9 &
#python vae.py --gpu=5 --dataset=color_count_10_1 --beta=$1 &
#python vae.py --gpu=6 --dataset=color_count_10_3 --beta=$1 &
#python vae.py --gpu=7 --dataset=color_count_10_5 --beta=$1 &
#python vae.py --gpu=4 --dataset=color_count_10_7 --beta=$1 &
#python vae.py --gpu=5 --dataset=color_count_10_9 --beta=$1 &



#python gan.py --gpu=5 --dataset=color_band_4_1 --model=gaussian_dcgan  &
#python gan.py --gpu=0 --dataset=color_band_4_3 --model=gaussian_dcgan  &
#python gan.py --gpu=1 --dataset=color_band_4_5 --model=gaussian_dcgan  &
#python gan.py --gpu=2 --dataset=color_band_4_7 --model=gaussian_dcgan  &
#python gan.py --gpu=3 --dataset=color_band_4_9 --model=gaussian_dcgan  &
#
#
#python gan.py --gpu=0 --dataset=color_band_4_1 --model=gaussian_wgan  &
#python gan.py --gpu=1 --dataset=color_band_4_3 --model=gaussian_wgan  &
#python gan.py --gpu=2 --dataset=color_band_4_5 --model=gaussian_wgan  &
#python gan.py --gpu=3 --dataset=color_band_4_7 --model=gaussian_wgan  &
#python gan.py --gpu=4 --dataset=color_band_4_9 --model=gaussian_wgan  &
#
#python gan.py --gpu=5 --dataset=color_random_4_1 --model=gaussian_wgan  &
#python gan.py --gpu=6 --dataset=color_random_4_3 --model=gaussian_wgan  &
#python gan.py --gpu=0 --dataset=color_random_4_5 --model=gaussian_wgan  &
#python gan.py --gpu=1 --dataset=color_random_4_7 --model=gaussian_wgan  &
#python gan.py --gpu=2 --dataset=color_random_4_9 --model=gaussian_wgan  &
#
#python gan.py --gpu=3 --dataset=color_band_4_1_9 --model=gaussian_wgan  &
#python gan.py --gpu=4 --dataset=color_band_4_3_7 --model=gaussian_wgan  &
#python gan.py --gpu=5 --dataset=color_band_4_1_3_5_7_9 --model=gaussian_wgan  &
#python gan.py --gpu=6 --dataset=color_band_4_1_5_9 --model=gaussian_wgan  &
#
##python gan.py --gpu=3 --dataset=color_random_4_1 --model=gaussian_dcgan  &
##python gan.py --gpu=4 --dataset=color_random_4_3 --model=gaussian_dcgan  &
##python gan.py --gpu=5 --dataset=color_random_4_5 --model=gaussian_dcgan  &
##python gan.py --gpu=0 --dataset=color_random_4_7 --model=gaussian_dcgan  &
##python gan.py --gpu=1 --dataset=color_random_4_9 --model=gaussian_dcgan  &
#
#
#python vae.py --gpu=6 --dataset=color_band_4_1 --beta=50.0  &
#python vae.py --gpu=7 --dataset=color_band_4_3 --beta=50.0  &
#python vae.py --gpu=6 --dataset=color_band_4_5 --beta=50.0  &
#python vae.py --gpu=7 --dataset=color_band_4_7 --beta=50.0  &
#python vae.py --gpu=6 --dataset=color_band_4_9 --beta=50.0  &
#python vae.py --gpu=7 --dataset=color_band_4_3_7 --beta=50.0  &
#
#python vae.py --gpu=7 --dataset=color_random_4_1 --beta=50.0  &
#python vae.py --gpu=6 --dataset=color_random_4_3 --beta=50.0  &
#python vae.py --gpu=7 --dataset=color_random_4_5 --beta=50.0  &
#python vae.py --gpu=6 --dataset=color_random_4_7 --beta=50.0  &
#python vae.py --gpu=7 --dataset=color_random_4_9 --beta=50.0  &

#python gan.py --gpu=0 --dataset=combi_46555 --model=gaussian_wgan  &
#
#python gan.py --gpu=1 --dataset=combi_44555 --model=gaussian_wgan  &
#python gan.py --gpu=2 --dataset=combi_45555 --model=gaussian_wgan  &
#python gan.py --gpu=3 --dataset=combi_47555 --model=gaussian_wgan  &
#
#python gan.py --gpu=4 --dataset=combi_46455 --model=gaussian_wgan  &
#python gan.py --gpu=5 --dataset=combi_46255 --model=gaussian_wgan  &
#python gan.py --gpu=6 --dataset=combi_46655 --model=gaussian_wgan  &
#python gan.py --gpu=7 --dataset=combi_46855 --model=gaussian_wgan  &
#
#python gan.py --gpu=0 --dataset=combi_46551 --model=gaussian_wgan  &

#python gan.py --gpu=1 --dataset=combi_46553 --model=gaussian_wgan  &
#python gan.py --gpu=2 --dataset=combi_46557 --model=gaussian_wgan  &
#python gan.py --gpu=3 --dataset=combi_46559 --model=gaussian_wgan  &
#python gan.py --gpu=0 --dataset=combi_46552 --model=gaussian_wgan  &
#python gan.py --gpu=1 --dataset=combi_46554 --model=gaussian_wgan  &
#python gan.py --gpu=2 --dataset=combi_46556 --model=gaussian_wgan  &
#python gan.py --gpu=3 --dataset=combi_46558 --model=gaussian_wgan  &

#
#python gan.py --gpu=4 --dataset=combi_46551_46553 --model=gaussian_wgan  &
#python gan.py --gpu=5 --dataset=combi_46555_46553 --model=gaussian_wgan  &
#python gan.py --gpu=6 --dataset=combi_46557_46553 --model=gaussian_wgan  &
#python gan.py --gpu=7 --dataset=combi_46559_46553 --model=gaussian_wgan  &

#python gan.py --gpu=4 --dataset=combi_46552_46553 --model=gaussian_wgan  &
#python gan.py --gpu=5 --dataset=combi_46554_46553 --model=gaussian_wgan  &
#python gan.py --gpu=6 --dataset=combi_46556_46553 --model=gaussian_wgan  &
#python gan.py --gpu=7 --dataset=combi_46558_46553 --model=gaussian_wgan  &

#python gan.py --gpu=0 --dataset=combi_45853_45857_45253_45257_47853_47857_47253_47257 --model=gaussian_wgan &
#python gan.py --gpu=1 --dataset=combi_45853_45253_47853_47253 --model=gaussian_wgan &
#python gan.py --gpu=2 --dataset=combi_45853_45857_45253_45257 --model=gaussian_wgan &
#python gan.py --gpu=3 --dataset=combi_47853_47857_47253_47257 --model=gaussian_wgan &




#python gan.py --gpu=0 --dataset=combif_44000 --model=gaussian_wgan  &
#python gan.py --gpu=1 --dataset=combif_45000 --model=gaussian_wgan  &
#python gan.py --gpu=2 --dataset=combif_46000 --model=gaussian_wgan  &
#python gan.py --gpu=3 --dataset=combif_47000 --model=gaussian_wgan  &
#
#python gan.py --gpu=4 --dataset=combif_40200 --model=gaussian_wgan  &
#python gan.py --gpu=5 --dataset=combif_40400 --model=gaussian_wgan  &
#python gan.py --gpu=6 --dataset=combif_40500 --model=gaussian_wgan  &
#python gan.py --gpu=7 --dataset=combif_40600 --model=gaussian_wgan  &
#python gan.py --gpu=0 --dataset=combif_40800 --model=gaussian_wgan  &
#
#python gan.py --gpu=1 --dataset=combif_40001 --model=gaussian_wgan  &
#python gan.py --gpu=2 --dataset=combif_40002 --model=gaussian_wgan  &
#python gan.py --gpu=3 --dataset=combif_40003 --model=gaussian_wgan  &
#python gan.py --gpu=4 --dataset=combif_40004 --model=gaussian_wgan  &
#python gan.py --gpu=5 --dataset=combif_40005 --model=gaussian_wgan  &
#python gan.py --gpu=6 --dataset=combif_40006 --model=gaussian_wgan  &
#python gan.py --gpu=7 --dataset=combif_40007 --model=gaussian_wgan  &
#python gan.py --gpu=0 --dataset=combif_40008 --model=gaussian_wgan  &
#python gan.py --gpu=1 --dataset=combif_40009 --model=gaussian_wgan  &

#python gan.py --gpu=2 --dataset=combif_40003_40004 --model=gaussian_wgan &
#python gan.py --gpu=3 --dataset=combif_40003_40005 --model=gaussian_wgan &
#python gan.py --gpu=4 --dataset=combif_40003_40006 --model=gaussian_wgan &
#python gan.py --gpu=5 --dataset=combif_40003_40007 --model=gaussian_wgan &
#python gan.py --gpu=6 --dataset=combif_40003_40008 --model=gaussian_wgan &
#python gan.py --gpu=7 --dataset=combif_40003_40009 --model=gaussian_wgan &


#python gan.py --gpu=2 --dataset=combif_40003_40004 --model=gaussian_wgan &
#python gan.py --gpu=3 --dataset=combif_40003_40005 --model=gaussian_wgan &
#python gan.py --gpu=4 --dataset=combif_40003_40006 --model=gaussian_wgan &
#python gan.py --gpu=5 --dataset=combif_40003_40007 --model=gaussian_wgan &
#python gan.py --gpu=6 --dataset=combif_40003_40008 --model=gaussian_wgan &
#python gan.py --gpu=7 --dataset=combif_40003_40009 --model=gaussian_wgan &

#python gan.py --gpu=2 --dataset=combif_40500_40600 --model=gaussian_wgan &
#python gan.py --gpu=2 --dataset=combif_40500_40700 --model=gaussian_wgan &
#python gan.py --gpu=2 --dataset=combif_40500_40800 --model=gaussian_wgan &
#python gan.py --gpu=2 --dataset=combif_40500_40400 --model=gaussian_wgan &

#python gan2.py --gpu=1 --dataset=crand_10 --model=gaussian_wgan --run=0 &
#python gan2.py --gpu=2 --dataset=crand_20 --model=gaussian_wgan --run=0 &
#python gan2.py --gpu=2 --dataset=crand_30 --model=gaussian_wgan --run=0 &
#python gan2.py --gpu=3 --dataset=crand_50 --model=gaussian_wgan --run=0 &
#python gan2.py --gpu=4 --dataset=crand_75 --model=gaussian_wgan --run=0 &
#python gan2.py --gpu=5 --dataset=crand_100 --model=gaussian_wgan --run=0 &
#python gan2.py --gpu=6 --dataset=crand_30 --model=gaussian_wgan --run=0 &
#python gan2.py --gpu=7 --dataset=crand_200 --model=gaussian_wgan --run=0 &
#
#python gan2.py --gpu=1 --dataset=crand_10 --model=gaussian_wgan --run=1 &
#python gan2.py --gpu=2 --dataset=crand_20 --model=gaussian_wgan --run=1 &
#python gan2.py --gpu=2 --dataset=crand_30 --model=gaussian_wgan --run=1 &
#python gan2.py --gpu=3 --dataset=crand_50 --model=gaussian_wgan --run=1 &
#python gan2.py --gpu=4 --dataset=crand_75 --model=gaussian_wgan --run=1 &
#python gan2.py --gpu=5 --dataset=crand_100 --model=gaussian_wgan --run=1 &
#python gan2.py --gpu=6 --dataset=crand_150 --model=gaussian_wgan --run=1 &
#python gan2.py --gpu=7 --dataset=crand_200 --model=gaussian_wgan --run=1 &

## Final experiment for prototypical enhancement
#python gan2.py --gpu=0 --dataset=cfixed_40003_40004 --model=gaussian_wgan --run=0 &
#python gan2.py --gpu=1 --dataset=cfixed_40003_40005 --model=gaussian_wgan --run=0 &
#python gan2.py --gpu=2 --dataset=cfixed_40003_40006 --model=gaussian_wgan --run=0 &
#python gan2.py --gpu=3 --dataset=cfixed_40003_40007 --model=gaussian_wgan --run=0 &
#python gan2.py --gpu=4 --dataset=cfixed_40003_40009 --model=gaussian_wgan --run=0 &
#
#python gan2.py --gpu=5 --dataset=cfixed_46553_46554 --model=gaussian_wgan --run=0 &
#python gan2.py --gpu=6 --dataset=cfixed_46553_46555 --model=gaussian_wgan --run=0 &
#python gan2.py --gpu=7 --dataset=cfixed_46553_46556 --model=gaussian_wgan --run=0 &
#python gan2.py --gpu=0 --dataset=cfixed_46553_46557 --model=gaussian_wgan --run=0 &
#python gan2.py --gpu=1 --dataset=cfixed_46553_46559 --model=gaussian_wgan --run=0 &
#

#python gan2.py --gpu=2 --dataset=cfixed_44553 --model=gaussian_wgan --run=0 &
#python gan2.py --gpu=3 --dataset=cfixed_44554 --model=gaussian_wgan --run=0 &
#python gan2.py --gpu=4 --dataset=cfixed_44555 --model=gaussian_wgan --run=0 &
#python gan2.py --gpu=5 --dataset=cfixed_44556 --model=gaussian_wgan --run=0 &
#python gan2.py --gpu=6 --dataset=cfixed_44557 --model=gaussian_wgan --run=0 &
#python gan2.py --gpu=7 --dataset=cfixed_44559 --model=gaussian_wgan --run=0 &
#
#python gan2.py --gpu=0 --dataset=cfixed_40003 --model=gaussian_wgan --run=0 &
#python gan2.py --gpu=1 --dataset=cfixed_40004 --model=gaussian_wgan --run=0 &
#python gan2.py --gpu=2 --dataset=cfixed_40005 --model=gaussian_wgan --run=0 &
#python gan2.py --gpu=3 --dataset=cfixed_40006 --model=gaussian_wgan --run=0 &
#python gan2.py --gpu=4 --dataset=cfixed_40007 --model=gaussian_wgan --run=0 &
#python gan2.py --gpu=5 --dataset=cfixed_40009 --model=gaussian_wgan --run=0 &