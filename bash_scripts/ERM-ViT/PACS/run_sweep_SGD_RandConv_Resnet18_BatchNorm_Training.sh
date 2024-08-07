#!/bin/bash

algorithms=New_RandConv_CNN
datasets=PACS # PACS, VLCS, OfficeHome, TerraIncognita, DomainNet, ImageNet_9 
data_dir=/media/SSD2/Dataset 
output_dir=./Results/${datasets}/RandConv_CNN_Clipped_loss_aug/Resnet18_New_RandConv_CNN_ks_1_3_dilation_2_4_clw_02_nonlinear_demix

for command in delete_incomplete launch
do
    python -m domainbed.scripts.sweep ${command} --data_dir=${data_dir} \
    --output_dir=${output_dir}  --command_launcher multi_gpu --algorithms ${algorithms}  \
    --single_domain_gen  --datasets ${datasets}  --n_hparams 1 --n_trials 2  \
    --hparams """{\"batch_size\":64,\"lr\":0.0001,\"kernel_size\":0.0,\"consistency_loss_w\":0.2,\"resnet_dropout\":0.0,\"val_augmentation\":true,\"invariant_loss\":false,\"invariant_loss_1\":true,\"alpha_min\":0.0,\"alpha_max\":1.0,\"weight_decay\":0.0,\"resnet18\":true,\"empty_fc\":true,\"fixed_featurizer\":false}"""
done

# nohup bash bash_scripts/ERM-ViT/PACS/run_sweep_SGD_RandConv_Resnet18_BatchNorm_Training.sh > OUT/PACS/RandConv_Summary/RandConv_CNN_Clipped_loss_aug_Resnet18_New_RandConv_CNN_ks_1_3_dilation_2_4_clw_02_nonlinear_demix.out 2>&1 &
