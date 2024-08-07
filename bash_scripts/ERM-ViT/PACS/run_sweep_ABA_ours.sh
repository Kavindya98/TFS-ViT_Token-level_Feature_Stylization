#!/bin/bash

algorithms=ABA_CNN
datasets=PACS # PACS, VLCS, OfficeHome, TerraIncognita, DomainNet, ImageNet_9 
data_dir=/media/SSD2/Dataset 
output_dir=./Results/${datasets}/ABA_CNN/ResNet18_64_PACS_ALT_ours_block_02

for command in delete_incomplete launch
do
    python -m domainbed.scripts.sweep ${command} --data_dir=${data_dir} \
    --output_dir=${output_dir}  --command_launcher multi_gpu3 --algorithms ${algorithms}  \
    --single_domain_gen  --datasets ${datasets}  --n_hparams 1 --n_trials 3  \
    --hparams """{\"num_blocks\":2,\"mixing\":false,\"clamp\":true,\"optim\":\"SGD\",\"scheduler\":false,\"pre_epoch\":4,\"elbo_beta\":0.1,\"lr_adv\":0.00005,\"batch_size\":64,\"lr\":0.0001,\"clw\":0.75,\"resnet_dropout\":0.0,\"weight_decay\":0.0,\"resnet18\":true,\"data_augmentation\":false,\"empty_fc\":true,\"fixed_featurizer\":false}"""
done

# nohup bash bash_scripts/ERM-ViT/PACS/run_sweep_ABA_ours.sh > OUT/PACS/ABA/ResNet18_64_PACS_ALT_ours_block_02.out 2>&1 & 