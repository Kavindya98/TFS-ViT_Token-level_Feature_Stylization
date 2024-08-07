#!/bin/bash

algorithms=New_RandConv_CNN
datasets=PACS # PACS, VLCS, OfficeHome, TerraIncognita, DomainNet, ImageNet_9 
data_dir=/media/SSD2/Dataset 
output_dir=./Results/${datasets}/RandConv_CNN_Clipped/ResNet_18

for command in delete_incomplete launch
do
    python -m domainbed.scripts.sweep ${command} --data_dir=${data_dir} \
    --output_dir=${output_dir}  --command_launcher multi_gpu --algorithms ${algorithms}  \
    --single_domain_gen  --datasets ${datasets}  --n_hparams 1 --n_trials 2  \
    --hparams """{\"batch_size\":32,\"lr\":0.0001,\"resnet_dropout\":0.0,\"weight_decay\":0.0,\"resnet18\":true,\"fixed_featurizer\":false,\"empty_fc\":true}"""
done
# nohup bash bash_scripts/ERM-ViT/PACS/run_sweep_SGD_RandConv.sh > PACS_randconv.out 2>&1 &