#!/bin/bash
algorithms=ERM
# datasets=ImageNet # PACS, VLCS, OfficeHome, TerraIncognita, DomainNet, ImageNet_9 
# data_dir=/media/SSD2/Dataset 
# output_dir=./Results/${datasets}/Fullset/Long_Epoch_Clipped/RandConv
# for command in delete_incomplete launch
# do
#     python -u -m domainbed.scripts.sweep ${command} --data_dir=${data_dir} \
#     --output_dir=${output_dir}  --command_launcher gpu_0 --algorithms ${algorithms}  \
#     --single_domain_gen  --datasets ${datasets}  --n_hparams 1 --n_trials 1  \
#     --hparams """{\"batch_size\":64,\"lr\":0.0001,\"resnet_dropout\":0.0,\"alpha_min\":0.0,\"alpha_max\":1.0,\"weight_decay\":0.0,\"custom_train_val\":true,\"custom_train\":0,\"custom_val\":1,\"resnet18\":false,\"fixed_featurizer\":false}"""
# done
#nohup bash bash_scripts/ERM-ViT/ImageNet/run_sweep_RandConv_CNN.sh > RandConv_Fixed_Feature_Clipped_No_Invariant.out 2>&1 &

# step 1 - train classifier from scratch
# datasets=ImageNet # PACS, VLCS, OfficeHome, TerraIncognita, DomainNet, ImageNet_9 
# data_dir=/media/SSD2/Dataset 
# output_dir=./Results/${datasets}/Fullset/Fixed_Feature_Clipped_No_Invariant/RandConv
# for command in delete_incomplete launch
# do
#     python -u -m domainbed.scripts.sweep ${command} --data_dir=${data_dir} \
#     --output_dir=${output_dir}  --command_launcher gpu_0 --algorithms ${algorithms}  \
#     --single_domain_gen  --datasets ${datasets}  --n_hparams 1 --n_trials 1  \
#     --hparams """{\"batch_size\":64,\"lr\":0.0001,\"identity_prob\":0.5,\"resnet_dropout\":0.0,\"invariant_loss\":false,\"consistency_loss_w\":0.0,\"alpha_min\":0.0,\"alpha_max\":1.0,\"weight_decay\":0.0,\"custom_train_val\":true,\"custom_train\":0,\"custom_val\":1,\"resnet18\":false,\"empty_fc\":true,\"fixed_featurizer\":true}"""
# done

# step 2 - finetuned with inconsistancy loss
datasets=ImageNet # PACS, VLCS, OfficeHome, TerraIncognita, DomainNet, ImageNet_9 
data_dir=/media/SSD2/Dataset 
output_dir=./Results/${datasets}/Fullset/ERM/ResNet50
for command in delete_incomplete launch
do
    python -u -m domainbed.scripts.sweep ${command} --data_dir=${data_dir} \
    --output_dir=${output_dir}  --command_launcher gpu_0 --algorithms ${algorithms}  \
    --single_domain_gen  --datasets ${datasets}  --n_hparams 1 --n_trials 1  \
    --hparams """{\"batch_size\":64,\"lr\":0.0001,\"resnet_dropout\":0.0,\"weight_decay\":0.0,\"custom_train_val\":true,\"custom_train\":0,\"custom_val\":1,\"resnet18\":false,\"empty_fc\":true,\"fixed_featurizer\":true}"""
done

# step 2.1 - finetuned without inconsistancy loss but with identity_prob 0
# nohup bash bash_scripts/ERM-ViT/ImageNet/run_sweep_ERM.sh > Fullset_ERM.out 2>&1 &