#!/bin/bash

algorithms=ERM_ViT
datasets=ImageNet_C # PACS, VLCS, OfficeHome, TerraIncognita, DomainNet, ImageNet_9 
backbone=DeitSmall # DeitSmall, T2T14# DeitSmall, T2T14
data_dir=/media/SSD2/Dataset
output_dir=./Results/${datasets}/ERM/DeitSmall_18_e_5

for command in delete_incomplete launch
do
    python -m domainbed.scripts.sweep ${command} --data_dir=${data_dir} \
    --output_dir=${output_dir}  --command_launcher gpu_2 --algorithms ${algorithms}  \
    --single_domain_gen  --datasets ${datasets}  --n_hparams 1 --n_trials 1  \
    --hparams """{\"backbone\":\"${backbone}\",\"batch_size\":32,\"lr\":5e-05 ,\"weight_decay\":1e-05,\"resnet_dropout\":0.0,\"fixed_featurizer\":false}"""
done
