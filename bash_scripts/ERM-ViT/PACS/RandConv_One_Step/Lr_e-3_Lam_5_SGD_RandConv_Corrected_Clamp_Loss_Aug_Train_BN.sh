#!/bin/bash

algorithms=RandConv_CNN
datasets=PACS # PACS, VLCS, OfficeHome, TerraIncognita, DomainNet, ImageNet_9 
data_dir=/media/SSD2/Dataset 
output_dir=./Results/${datasets}/RandConv_One_Step/Lr_e-3_Lam_5_SGD_RandConv_Corrected_Clamp_Loss_Aug_Train_BN
for command in delete_incomplete launch
do
    python -m domainbed.scripts.sweep ${command} --data_dir=${data_dir} \
    --output_dir=${output_dir}  --command_launcher multi_gpu --algorithms ${algorithms}  \
    --single_domain_gen  --datasets ${datasets}  --n_hparams 1 --n_trials 3  \
    --hparams """{\"batch_size\":64,\"lr\":0.0001,\"kernel_size\":0.0,\"consistency_loss_w\":5.0,\"resnet_dropout\":0.0,\"val_augmentation\":true,\"alpha_min\":0.0,\"alpha_max\":1.0,\"weight_decay\":0.0,\"custom_train_val\":true,\"custom_train\":0,\"custom_val\":1,\"resnet18\":true,\"empty_fc\":true,\"fixed_featurizer\":false}"""
done

# nohup bash bash_scripts/ERM-ViT/PACS/RandConv_One_Step/Lr_e-3_Lam_5_SGD_RandConv_Corrected_Clamp_Loss_Aug_Train_BN.sh > OUT/PACS/RandConv_Summary/Lr_e-3_Lam_5_SGD_RandConv_Corrected_Clamp_Loss_Aug_Train_BN.out 2>&1 &