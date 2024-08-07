#!/bin/bash

algorithms=ADA_ViT
datasets=PACS_Custom # PACS, VLCS, OfficeHome, TerraIncognita, DomainNet, ImageNet_9 
data_dir=/media/SSD2/Dataset 
output_dir=./Results/${datasets}/ADA_ViT/T2T14_64_PACS_unsplit_0_workers_2

for command in delete_incomplete launch
do
    for backbone in "T2T14"  
    do 
        python -m domainbed.scripts.sweep ${command} --data_dir=${data_dir} \
        --output_dir=${output_dir}  --command_launcher multi_gpu --algorithms ${algorithms}  \
        --single_domain_gen  --datasets ${datasets}  --n_hparams 1 --n_trials 3  \
        --hparams """{\"backbone\":\"${backbone}\",\"batch_size\":64,\"lr\":5e-05 ,\"resnet_dropout\":0.0,\"weight_decay\":0.0,\"fixed_featurizer\":false,\"empty_head\":true}"""
    done
done

# nohup bash bash_scripts/ERM-ViT/PACS/run_sweep_ADA_ViT.sh > OUT/PACS/ADA/T2T14_64_PACS_unsplit_2.out 2>&1 &