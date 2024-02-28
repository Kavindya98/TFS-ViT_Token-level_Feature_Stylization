#!/bin/bash

algorithms=AugMix_CNN
datasets=PACS # PACS, VLCS, OfficeHome, TerraIncognita, DomainNet, ImageNet_9 
data_dir=/media/SSD2/Dataset 
output_dir=./Results/${datasets}/AugMix_ViT/ResNet18
for command in delete_incomplete launch
do
    python -m domainbed.scripts.sweep ${command} --data_dir=${data_dir} \
    --output_dir=${output_dir}  --command_launcher multi_gpu --algorithms ${algorithms}  \
    --single_domain_gen  --datasets ${datasets}  --n_hparams 1 --n_trials 3  \
    --hparams """{\"batch_size\":64,\"val_augmentation\":true,\"consistency_loss_w\":12.0,\"normalization\":false,\"lr\":0.0001,\"resnet_dropout\":0.0,\"weight_decay\":0.0,\"resnet18\":true,\"data_augmentation\":false,\"empty_fc\":true,\"fixed_featurizer\":false}"""
done

# nohup bash bash_scripts/ERM-ViT/PACS/run_sweep_AugMix_ResNet18.sh > OUT/PACS/AugMix_ViT/Resnet18.ou 2>&1 &