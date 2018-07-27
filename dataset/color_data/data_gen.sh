#!/usr/bin/env bash

#python2 dataset_generator.py --bn=0 --bs=8192 --type=$1 &
#python2 dataset_generator.py --bn=1 --bs=8192 --type=$1 &
#python2 dataset_generator.py --bn=2 --bs=8192 --type=$1 &
#python2 dataset_generator.py --bn=3 --bs=8192 --type=$1 &
#python2 dataset_generator.py --bn=4 --bs=8192 --type=$1 &
#python2 dataset_generator.py --bn=5 --bs=8192 --type=$1 &
#python2 dataset_generator.py --bn=6 --bs=8192 --type=$1 &
#python2 dataset_generator.py --bn=7 --bs=8192 --type=$1 &


python2 dataset_generator_combi.py --bn=0 --bs=8192 --type=$1 &
python2 dataset_generator_combi.py --bn=1 --bs=8192 --type=$1 &
python2 dataset_generator_combi.py --bn=2 --bs=8192 --type=$1 &
python2 dataset_generator_combi.py --bn=3 --bs=8192 --type=$1 &
python2 dataset_generator_combi.py --bn=4 --bs=8192 --type=$1 &
python2 dataset_generator_combi.py --bn=5 --bs=8192 --type=$1 &
python2 dataset_generator_combi.py --bn=6 --bs=8192 --type=$1 &
python2 dataset_generator_combi.py --bn=7 --bs=8192 --type=$1 &


#python2 dataset_generator_combif.py --bn=0 --bs=25000 --configs=$1 &
#python2 dataset_generator_combif.py --bn=1 --bs=25000 --configs=$1 &
#python2 dataset_generator_combif.py --bn=2 --bs=25000 --configs=$1 &
#python2 dataset_generator_combif.py --bn=3 --bs=25000 --configs=$1 &
#python2 dataset_generator_combif.py --bn=4 --bs=25000 --configs=$1 &
#python2 dataset_generator_combif.py --bn=5 --bs=25000 --configs=$1 &
#python2 dataset_generator_combif.py --bn=6 --bs=25000 --configs=$1 &
#python2 dataset_generator_combif.py --bn=7 --bs=25000 --configs=$1 &
