#!/bin/bash

GPU=0
seed=31  # 31 32 33 34
code=main_imp.py 

data_dir=./data/
dataset=MNIST_75sp_0.8 # MNIST_75sp_0.9 MNIST_75sp_0.95 fashion_0.8 fashion_0.9 fashion_0.95 kuzu_biased_0.8 kuzu_biased_0.9 kuzu_biased_0.95
all_epochs=200
use_mask=1
swap_epochs=100
q=0.7
lambda_dis=1
lambda1=15
lambda3=1
if echo "$dataset" | grep -q "^MNIST.*"; then
    lambda2=0.01
else
    lambda2=0.05
fi

out_dir="output_GCN_"$dataset"_seed_"$seed"_lambda1_"$lambda1"_lambda2_"$lambda2"_lambda3_"$lambda3""
python -u $code --config 'configs/superpixels_graph_classification_GPro_MNIST_100k.json' \
--dataset $dataset \
--data_dir $data_dir \
--seed $seed  \
--mask_epochs $all_epochs \
--swap_epochs $swap_epochs \
--lambda1 $lambda1 \
--lambda2 $lambda2 \
--lambda3 $lambda3 \
--use_mask $use_mask \
--q $q \
--lambda_dis $lambda_dis \
--out_dir $out_dir \
--gpu_id $GPU

