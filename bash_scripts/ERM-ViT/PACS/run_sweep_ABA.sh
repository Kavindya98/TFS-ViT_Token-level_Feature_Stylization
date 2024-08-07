#!/bin/bash

algorithms=ABA_CNN
datasets=PACS # PACS, VLCS, OfficeHome, TerraIncognita, DomainNet, ImageNet_9 
data_dir=/media/SSD2/Dataset 
output_dir=./Results/${datasets}/ABA_CNN/ResNet50_64_PACS_ALT_original

for command in delete_incomplete launch
do
    python -m domainbed.scripts.sweep ${command} --data_dir=${data_dir} \
    --output_dir=${output_dir}  --command_launcher multi_gpu --algorithms ${algorithms}  \
    --single_domain_gen  --datasets ${datasets}  --n_hparams 1 --n_trials 1  \
    --hparams """{\"num_blocks\":0,\"mixing\":false,\"clamp\":true,\"optim\":\"SGD\",\"scheduler\":true,\"pre_epoch\":4,\"elbo_beta\":0.1,\"lr_adv\":0.00005,\"batch_size\":64,\"lr\":0.004,\"clw\":0.75,\"resnet_dropout\":0.0,\"weight_decay\":0.0,\"resnet18\":false,\"data_augmentation\":false,\"empty_fc\":true,\"fixed_featurizer\":false}"""
done

# nohup bash bash_scripts/ERM-ViT/PACS/run_sweep_ABA.sh > OUT/PACS/ABA/ResNet50_64_PACS_ALT_original.out 2>&1 & 