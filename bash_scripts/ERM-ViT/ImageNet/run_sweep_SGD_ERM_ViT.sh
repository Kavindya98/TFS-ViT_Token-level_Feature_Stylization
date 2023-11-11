#!/bin/bash

algorithms=RandConv_ViT
datasets=ImageNet # PACS, VLCS, OfficeHome, TerraIncognita, DomainNet, ImageNet_9 
#backbone=DeiTBase ViTBase  # DeitSmall, T2T14# DeitSmall, T2T14
data_dir=/media/SSD2/Dataset
output_dir=./Results/${datasets}/RandConv_ViT/${backbone}

for command in delete_incomplete launch
do
    for backbone in ViTBase 
    do 
        output_dir=./Results/${datasets}/RandConv_ViT/${backbone}_1
        
        python -m domainbed.scripts.sweep ${command} --data_dir=${data_dir} \
        --output_dir=${output_dir}  --command_launcher multi_gpu --algorithms ${algorithms}  \
        --single_domain_gen  --datasets ${datasets}  --n_hparams 1 --n_trials 2  \
        --hparams """{\"backbone\":\"${backbone}\",\"batch_size\":32,\"lr\":5e-05 ,\"resnet_dropout\":0.0,\"custom_train_val\":true,\"custom_train\":0,\"custom_val\":1,\"weight_decay\":0.0,\"fixed_featurizer\":false}"""
    done
done
