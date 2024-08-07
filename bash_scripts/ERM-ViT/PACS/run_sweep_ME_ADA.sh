#!/bin/bash

algorithms=ME_ADA_CNN
datasets=PACS_Custom # PACS, VLCS, OfficeHome, TerraIncognita, DomainNet, ImageNet_9 
data_dir=/media/SSD2/Dataset 
output_dir=./Results/${datasets}/ME_ADA_CNN/ResNet50_64_PACS_unsplit_corrected

for command in delete_incomplete launch
do
    python -m domainbed.scripts.sweep ${command} --data_dir=${data_dir} \
    --output_dir=${output_dir}  --command_launcher multi_gpu --algorithms ${algorithms}  \
    --single_domain_gen  --datasets ${datasets}  --n_hparams 1 --n_trials 3  \
    --hparams """{\"batch_size\":64,\"lr\":0.0001,\"resnet_dropout\":0.0,\"weight_decay\":0.0,\"resnet18\":false,\"data_augmentation\":false,\"empty_fc\":true,\"fixed_featurizer\":false}"""
done

# nohup bash bash_scripts/ERM-ViT/PACS/run_sweep_ME_ADA.sh > OUT/PACS/ME_ADA/ResNet50_64_PACS_unsplit_corrected.out 2>&1 &