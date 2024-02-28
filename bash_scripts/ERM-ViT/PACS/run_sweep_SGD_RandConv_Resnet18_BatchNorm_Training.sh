#!/bin/bash

algorithms=RandConv_CNN
datasets=PACS # PACS, VLCS, OfficeHome, TerraIncognita, DomainNet, ImageNet_9 
data_dir=/media/SSD2/Dataset 
output_dir=./Results/${datasets}/RandConv_CNN_Clipped_loss_aug/With_BNN/Resnet18

for command in delete_incomplete launch
do
    python -m domainbed.scripts.sweep ${command} --data_dir=${data_dir} \
    --output_dir=${output_dir}  --command_launcher multi_gpu --algorithms ${algorithms}  \
    --single_domain_gen  --datasets ${datasets}  --n_hparams 1 --n_trials 3  \
    --hparams """{\"batch_size\":64,\"lr\":0.0001,\"kernel_size\":0.0,\"consistency_loss_w\":10.0,\"resnet_dropout\":0.0,\"val_augmentation\":true,\"alpha_min\":0.0,\"alpha_max\":1.0,\"weight_decay\":0.0,\"resnet18\":true,\"empty_fc\":true,\"fixed_featurizer\":false}"""
done

# nohup bash bash_scripts/ERM-ViT/PACS/run_sweep_SGD_RandConv_Resnet18_BatchNorm_Training.sh > OUT/PACS/RandConv_Summary/RandConv_CNN_Clipped_loss_aug_With_BNN_Resnet18.out 2>&1 &
