#!/bin/bash

algorithms=RandConv_CNN
datasets=PACS # PACS, VLCS, OfficeHome, TerraIncognita, DomainNet, ImageNet_9 
data_dir=/media/SSD2/Dataset 
output_dir=./Results/${datasets}/ABA/RandConv_CNN

for command in delete_incomplete launch
do
    python -m domainbed.scripts.sweep ${command} --data_dir=${data_dir} \
    --output_dir=${output_dir}  --command_launcher multi_gpu --algorithms ${algorithms}  \
    --single_domain_gen  --datasets ${datasets}  --n_hparams 1 --n_trials 3  \
    --hparams """{\"optim\":\"SGD\",\"batch_size\":32,\"lr\":0.0004,\"resnet_dropout\":0.0,\"weight_decay\":0.0,\"scheduler\":true,\"resnet18\":false,\"data_augmentation\":false,\"empty_fc\":true,\"fixed_featurizer\":false}"""
done

# nohup bash bash_scripts/ERM-ViT/PACS/run_sweep_SGD_ERM_ABA.sh > OUT/PACS/ABA/RandConv_CNN.out 2>&1 &