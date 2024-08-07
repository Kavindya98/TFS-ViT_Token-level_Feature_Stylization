#!/bin/bash

algorithms=RandConv_ViT
datasets=PACS # PACS, VLCS, OfficeHome, TerraIncognita, DomainNet, ImageNet_9 
#backbone=ViTBase # DeitSmall, T2T14# DeitSmall, T2T14
data_dir=/media/SSD2/Dataset
#output_dir=./Results/${datasets}/ERM_VIT/DeitBase

for command in delete_incomplete launch
do
    for backbone in DeitSmall 
    do 
        output_dir=./Results/${datasets}/RandConv_ViT_Clipped/${backbone}
        python -u -m domainbed.scripts.sweep ${command} --data_dir=${data_dir} \
        --output_dir=${output_dir}  --command_launcher multi_gpu --algorithms ${algorithms}  \
        --single_domain_gen  --datasets ${datasets}  --n_hparams 1 --n_trials 3  \
        --hparams """{\"backbone\":\"${backbone}\",\"batch_size\":64,\"lr\":5e-05 ,\"val_augmentation\":true,\"resnet_dropout\":0.0,\"weight_decay\":0.0,\"fixed_featurizer\":false,\"empty_head\":true}"""
    done
done

# nohup bash bash_scripts/ERM-ViT/PACS/run_sweep_SGD_Randconv_ViT.sh > PACS_randconv_VIT.out 2>&1 &
