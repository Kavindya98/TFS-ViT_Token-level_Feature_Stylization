#!/bin/bash

algorithms=ERM_ViT
datasets=PACS # PACS, VLCS, OfficeHome, TerraIncognita, DomainNet
backbone=DeitSmall # DeitSmall, T2T14
data_dir=/home/kavindya/data/Models/SDViT/domainbed/data
output_dir=./Results/PACS/ERM_ViT/DeitSmall

for command in delete_incomplete launch
do
    python -m domainbed.scripts.sweep ${command} --data_dir=${data_dir} \
    --output_dir=${output_dir}  --command_launcher multi_gpu --algorithms ${algorithms}  \
    --single_domain_gen  --datasets ${datasets}  --n_hparams 1 --n_trials 1  \
    --hparams """{\"backbone\":\"${backbone}\",\"batch_size\":32,\"lr\":5e-05,\"resnet_dropout\":0.0,\"weight_decay\":0.0}"""
done
