#!/bin/bash
algorithms=RandConv_CNN
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
#nohup bash bash_scripts/ERM-ViT/ImageNet/run_sweep_RandConv_CNN.sh > SGD_Fixed_Feature_Clipped_No_Invariant.out 2>&1 &

# step 1 - train classifier from scratch
datasets=ImageNet # PACS, VLCS, OfficeHome, TerraIncognita, DomainNet, ImageNet_9 
data_dir=/media/SSD2/Dataset 
output_dir=./Results/${datasets}/Fullset/SGD_Fixed_Feature_Clipped_No_Invariant/RandConv
for command in delete_incomplete launch
do
    python -u -m domainbed.scripts.sweep ${command} --data_dir=${data_dir} \
    --output_dir=${output_dir}  --command_launcher gpu_0 --algorithms ${algorithms}  \
    --single_domain_gen  --datasets ${datasets}  --n_hparams 1 --n_trials 1  \
    --hparams """{\"batch_size\":64,\"lr\":0.001,\"kernel_size\":0.0,\"identity_prob\":0.5,\"resnet_dropout\":0.0,\"val_augmentation\":true,\"invariant_loss\":false,\"consistency_loss_w\":0.0,\"alpha_min\":0.0,\"alpha_max\":1.0,\"weight_decay\":0.0,\"custom_train_val\":true,\"custom_train\":0,\"custom_val\":1,\"resnet18\":false,\"empty_fc\":true,\"fixed_featurizer\":true}"""
done

# # step 2 - finetuned with inconsistancy loss
# datasets=ImageNet # PACS, VLCS, OfficeHome, TerraIncognita, DomainNet, ImageNet_9 
# data_dir=/media/SSD2/Dataset 
# output_dir=./Results/${datasets}/Fullset/2_1_Full_Model_Clipped_IDP_0_No_Invariant/RandConv
# checkpoint_URL=./Results/ImageNet/Fullset/Fixed_Feature_Clipped_No_Invariant/RandConv/t1_s2/IID_best.pkl
# for command in delete_incomplete launch
# do
#     python -u -m domainbed.scripts.sweep ${command} --data_dir=${data_dir} \
#     --output_dir=${output_dir}  --command_launcher gpu_1 --algorithms ${algorithms}  \
#     --single_domain_gen  --datasets ${datasets}  --n_hparams 1 --n_trials 1 --continue_checkpoint ${checkpoint_URL} \
#     --hparams """{\"batch_size\":64,\"lr\":0.0001,\"checkpoint_step_start\":40000,\"resnet_dropout\":0.0,\"consistency_loss_w\":0.0,\"invariant_loss\":false,\"alpha_min\":0.0,\"alpha_max\":1.0,\"weight_decay\":0.0,\"custom_train_val\":true,\"custom_train\":0,\"custom_val\":1,\"resnet18\":false,\"empty_fc\":true,\"fixed_featurizer\":false}"""
# done

# step 2.1 - finetuned without inconsistancy loss but with identity_prob 0
# nohup bash bash_scripts/ERM-ViT/ImageNet/run_sweep_RandConv_CNN.sh > 2_1_Full_Model_Clipped_IDP_0_No_Invariant.out 2>&1 &

# step 1.1 - train classifier with high kernel size
# datasets=ImageNet # PACS, VLCS, OfficeHome, TerraIncognita, DomainNet, ImageNet_9 
# data_dir=/media/SSD2/Dataset 
# output_dir=./Results/${datasets}/Fullset/High_Kernel_Fixed_Feature_Clipped_No_Invariant/RandConv
# for command in delete_incomplete launch
# do
#     python -u -m domainbed.scripts.sweep ${command} --data_dir=${data_dir} \
#     --output_dir=${output_dir}  --command_launcher gpu_0 --algorithms ${algorithms}  \
#     --single_domain_gen  --datasets ${datasets}  --n_hparams 1 --n_trials 1  \
#     --hparams """{\"batch_size\":64,\"lr\":0.0001,\"kernel_size\":3.0,\"identity_prob\":0.5,\"resnet_dropout\":0.0,\"invariant_loss\":false,\"consistency_loss_w\":0.0,\"alpha_min\":0.0,\"alpha_max\":1.0,\"weight_decay\":0.0,\"custom_train_val\":true,\"custom_train\":0,\"custom_val\":1,\"resnet18\":false,\"empty_fc\":true,\"fixed_featurizer\":true}"""
# done