#!/bin/bash

algorithms=ABA_ViT
datasets=PACS # PACS, VLCS, OfficeHome, TerraIncognita, DomainNet, ImageNet_9 
data_dir=/media/SSD2/Dataset 
backbone=T2T14
output_dir=./Results/${datasets}/ABA_ViT/T2T14_64_PACS_our_elbo_1_lr_5e4_ALT

for command in delete_incomplete launch
do
    python -m domainbed.scripts.sweep ${command} --data_dir=${data_dir} \
    --output_dir=${output_dir}  --command_launcher multi_gpu --algorithms ${algorithms}  \
    --single_domain_gen  --datasets ${datasets}  --n_hparams 1 --n_trials 3  \
    --hparams """{\"num_blocks\":0,\"backbone\":\"${backbone}\",\"mixing\":false,\"clamp\":true,\"pre_epoch\":4,\"elbo_beta\":1,\"lr_adv\":0.0005,\"batch_size\":64,\"lr\":5e-05,\"clw\":0.75,\"resnet_dropout\":0.0,\"weight_decay\":0.0,\"data_augmentation\":false,\"empty_fc\":true,\"fixed_featurizer\":false}"""
done

# nohup bash bash_scripts/ERM-ViT/PACS/run_sweep_ABA_ViT.sh > OUT/PACS/ABA/T2T14_64_PACS_our_elbo_1_lr_5e4_ALT.out 2>&1 & 