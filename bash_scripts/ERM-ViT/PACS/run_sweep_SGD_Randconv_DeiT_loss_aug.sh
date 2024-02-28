#!/bin/bash

algorithms=RandConv_ViT
datasets=PACS # PACS, VLCS, OfficeHome, TerraIncognita, DomainNet, ImageNet_9 
#backbone=ViTBase # DeitSmall, T2T14# DeitSmall, T2T14
data_dir=/media/SSD2/Dataset
#output_dir=./Results/${datasets}/ERM_VIT/DeitBase
for backbone in DeiTBase ViTBase

do
for consistency_loss in 10 #5
do
for kernel_size in 0 #2
do
for alpha_min in 0.0 #0.5
do
    for command in delete_incomplete launch
    do 
        output_dir=./Results/${datasets}/RandConv_ViT_Clipped_loss_aug/${backbone}
        python -u -m domainbed.scripts.sweep ${command} --data_dir=${data_dir} \
        --output_dir=${output_dir}_cl_${consistency_loss}_ks_${kernel_size}_al_${alpha_min}  --command_launcher multi_gpu --algorithms ${algorithms}  \
        --single_domain_gen  --datasets ${datasets}  --n_hparams 1 --n_trials 3  \
        --hparams """{\"backbone\":\"${backbone}\",\"kernel_size\":\"${kernel_size}\",\"consistency_loss_w\":\"${consistency_loss}\",\"loss_aug\":true,\"alpha_min\":\"${alpha_min}\",\"alpha_max\":1.0,\"batch_size\":64,\"lr\":5e-05 ,\"resnet_dropout\":0.0,\"val_augmentation\":true,\"weight_decay\":0.0,\"fixed_featurizer\":false,\"empty_head\":true}"""
    done
done
done
done
done

# nohup bash bash_scripts/ERM-ViT/PACS/run_sweep_SGD_Randconv_DeiT_loss_aug.sh > OUT/PACS/RandConv_Summary/DeiT_ViT_Base_T2T14_loss_aug.out 2>&1 &
