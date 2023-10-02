#!/bin/bash

algorithms=ERM_ViT
datasets=ImageNet_9 # PACS, VLCS, OfficeHome, TerraIncognita, DomainNet, ImageNet_9 
backbone=DeitSmall # DeitSmall, T2T14# DeitSmall, T2T14
data_dir=/media/SSD2/Dataset
output_dir=./Results/${datasets}/ERM/DeitSmall

for command in delete_incomplete launch
do
    python -m domainbed.scripts.sweep ${command} --data_dir=${data_dir} \
    --output_dir=${output_dir}  --command_launcher gpu_4 --algorithms ${algorithms}  \
    --single_domain_gen  --datasets ${datasets}  --n_hparams 1 --n_trials 1  \
    --hparams """{\"backbone\":\"${backbone}\",\"batch_size\":32,\"lr\":5e-05 ,\"resnet_dropout\":0.0,\"weight_decay\":0.0,\"fixed_featurizer\":false}"""
done
