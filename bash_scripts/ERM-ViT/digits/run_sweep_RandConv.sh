#!/bin/bash

algorithms=RandConv_CNN
datasets=DIGITS # PACS, VLCS, OfficeHome, TerraIncognita, DomainNet, ImageNet_9 
data_dir=/media/SSD2/Dataset 
output_dir=./Results/${datasets}/ABA_CNN/RandConv_CNN

for command in delete_incomplete launch
do
    python -u  -m domainbed.scripts.sweep ${command} --data_dir=${data_dir} \
    --output_dir=${output_dir}  --command_launcher multi_gpu --algorithms ${algorithms}  \
    --single_domain_gen  --datasets ${datasets}  --n_hparams 1 --n_trials 3  \
    --hparams """{\"batch_size\":512,\"lr\":0.0001,\"optim\":\"Adam\",\"custom_train_val\":true,\"custom_train\":0,\"custom_val\":1,\"resnet_dropout\":0.0,\"weight_decay\":0.0,\"empty_fc\":true,\"fixed_featurizer\":false}"""
done

# nohup bash bash_scripts/ERM-ViT/digits/run_sweep_RandConv.sh > OUT/DIGITS/ABA/RandConv_CNN.out 2>&1 &