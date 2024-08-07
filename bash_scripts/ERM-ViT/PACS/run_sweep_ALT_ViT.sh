#!/bin/bash

algorithms=ALT_ViT
datasets=PACS # PACS, VLCS, OfficeHome, TerraIncognita, DomainNet, ImageNet_9 
data_dir=/media/SSD2/Dataset 
backbone=DeiTSmall
output_dir=./Results/${datasets}/ALT_ViT/DeiTSmall_64_PACS_ALT_original_block_0_5e5_2

for command in delete_incomplete launch
do
    python -m domainbed.scripts.sweep ${command} --data_dir=${data_dir} \
    --output_dir=${output_dir}  --command_launcher multi_gpu --algorithms ${algorithms}  \
    --single_domain_gen  --datasets ${datasets}  --n_hparams 1 --n_trials 3  \
    --hparams """{\"num_blocks\":0,\"backbone\":\"${backbone}\",\"mixing\":false,\"pre_epoch\":4,\"lr_adv\":0.00005,\"batch_size\":64,\"lr\":5e-05,\"clw\":0.75,\"resnet_dropout\":0.0,\"weight_decay\":0.0,\"data_augmentation\":false,\"fixed_featurizer\":false,\"empty_head\":true}"""
done

# nohup bash bash_scripts/ERM-ViT/PACS/run_sweep_ALT_ViT.sh > OUT/PACS/ALT/DeiTSmall_64_PACS_ALT_original_block_0_5e5_2.out 2>&1 & 