#!/bin/bash

algorithms=ABA_CNN
datasets=PACS # PACS, VLCS, OfficeHome, TerraIncognita, DomainNet, ImageNet_9 
data_dir=/media/SSD2/Dataset 
output_dir=./Results/${datasets}/ABA_CNN/ResNet18_32_PACS_unsplit_num_blocks_0_mixing_t_clamp_t_lr_adv_corrected

for command in delete_incomplete launch
do
    python -m domainbed.scripts.sweep ${command} --data_dir=${data_dir} \
    --output_dir=${output_dir}  --command_launcher multi_gpu --algorithms ${algorithms}  \
    --single_domain_gen  --datasets ${datasets}  --n_hparams 1 --n_trials 3  \
    --hparams """{\"num_blocks\":0,\"mixing\":false,\"clamp\":true,\"elbo_beta\":0.1,\"lr_adv\":0.00005,\"batch_size\":32,\"lr\":0.0004,\"resnet_dropout\":0.0,\"weight_decay\":0.0,\"resnet18\":true,\"data_augmentation\":false,\"scheduler\":true,\"empty_fc\":true,\"fixed_featurizer\":false}"""
done

# nohup bash bash_scripts/ERM-ViT/PACS/run_sweep_ABA_2.sh > OUT/PACS/ABA/ResNet18_32_PACS_unsplit_num_blocks_0_mixing_t_clamp_t_lr_adv_corrected.out 2>&1 &