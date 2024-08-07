#!/bin/bash

algorithms=TFSViT
alpha=0.1
datasets=PACS # PACS, VLCS, OfficeHome, TerraIncognita, DomainNet
backbone=DeiTBase # DeitSmall, T2T14
data_dir=/media/SSD2/Dataset
output_dir=./Results/${datasets}/unsplit/${algorithms}/${backbone}


for n_layers in 1 #2 3 4  # number of random layers to apply TFS (n in the paper) 1 2 4
do
    for d_rate in 0.1 #0.3 0.5 0.8 # the rate of token selection and replacement (d in the paper) 0.3 0.5 0.8
        do
        for command in delete_incomplete launch
        do
            python -m domainbed.scripts.sweep ${command} --data_dir=${data_dir} \
            --output_dir=${output_dir}/sweep_drate_${d_rate}_nlay_${n_layers}  --command_launcher multi_gpu --algorithms ${algorithms}  \
            --single_domain_gen  --datasets ${datasets}  --n_hparams 1 --n_trials 3  \
            --hparams """{\"backbone\":\"${backbone}\",\"batch_size\":64,\"lr\":5e-05,\"resnet_dropout\":0.0,\"weight_decay\":0.0,\"fixed_featurizer\":false,\"empty_head\":true,\"num_layers\":$n_layers,\"d_rate\":$d_rate,\"alpha\":$alpha}"""
        done
    done
done

# for command in delete_incomplete launch
#     do
#         python -m domainbed.scripts.sweep ${command} --data_dir=${data_dir} \
#         --output_dir=${output_dir}/sweep_drate_${d_rate}_nlay_${n_layers}  --command_launcher multi_gpu --algorithms ${algorithms}  \
#         --single_domain_gen  --datasets ${datasets}  --n_hparams 1 --n_trials 1  \
#         --hparams """{\"backbone\":\"${backbone}\",\"batch_size\":64,\"lr\":5e-05,\"resnet_dropout\":0.0,\"custom_train_val\":true,\"custom_train\":0,\"custom_val\":1,\"weight_decay\":0.0,\"fixed_featurizer\":false,\"num_layers\":3,\"d_rate\":0.3,\"alpha\":$alpha}"""
#     done

# nohup bash bash_scripts/TFS-ViT/run_sweep.sh > OUT/PACS/TFSViT/DeiTBase_PACS_Unsplit_All_diff_val_aug.out 2>&1 &