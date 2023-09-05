#!/bin/bash

algorithms=ERM
datasets=DomainNet # PACS, VLCS, OfficeHome, TerraIncognita, DomainNet # DeitSmall, T2T14
data_dir=/home/kavindya/data/Models/DATA
output_dir=./Results/${datasets}/ERM/ResNet_50
/
for command in delete_incomplete launch
do
    python -m domainbed.scripts.sweep ${command} --data_dir=${data_dir} \
    --output_dir=${output_dir}  --command_launcher gpu_1 --algorithms ${algorithms}  \
    --single_domain_gen  --datasets ${datasets}  --n_hparams 1 --n_trials 1  \
    --hparams """{\"batch_size\":32,\"lr\":0.0001,\"resnet_dropout\":0.0,\"weight_decay\":0.0,\"resnet18\":false}"""
done
