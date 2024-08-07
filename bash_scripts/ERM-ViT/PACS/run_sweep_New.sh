#!/bin/bash

algorithms=New_CNN
datasets=PACS # PACS, VLCS, OfficeHome, TerraIncognita, DomainNet, ImageNet_9 
data_dir=/media/SSD2/Dataset 
output_dir=./Results/${datasets}/New_CNN/ResNet18_different_croos_entropy_severity_1_no_AugMix_clw_1_different_random_clamped
for command in delete_incomplete launch
do
    python -m domainbed.scripts.sweep ${command} --data_dir=${data_dir} \
    --output_dir=${output_dir}  --command_launcher multi_gpu3 --algorithms ${algorithms}  \
    --single_domain_gen  --datasets ${datasets}  --n_hparams 1 --n_trials 3  \
    --hparams """{\"batch_size\":64,\"val_augmentation\":true,\"lr_adv\":0.1,\"pre_epoch\":5,\"adv_steps\":7,\"with_AugMix\":false,\"consistency_loss_w\":1,\"normalization\":false,\"lr\":0.0001,\"optim\":\"SGD\",\"resnet_dropout\":0.0,\"weight_decay\":0.0,\"resnet18\":true,\"data_augmentation\":false,\"empty_fc\":true,\"fixed_featurizer\":false}"""
done

# nohup bash bash_scripts/ERM-ViT/PACS/run_sweep_New.sh > OUT/PACS/New_CNN/ResNet18_different_croos_entropy_severity_1_no_AugMix_clw_1_different_random_clamped.out 2>&1 &