algorithms=RandConv_CNN
datasets=ImageNet # PACS, VLCS, OfficeHome, TerraIncognita, DomainNet, ImageNet_9 
data_dir=/media/SSD2/Dataset 
output_dir=./Results/${datasets}/Fullset/SGD_ReNet/RandConv/RandConv_Second/Lam_5_SGD_Val_Aug_BN_Only_Clipped
checkpoint_URL=./Results/ImageNet/Fullset/SGD_ReNet/RandConv/Val_Aug_Fixed_Feature_Clipped_No_Invariant/t1_s2/IID_best.pkl
for command in delete_incomplete launch
do
    python -u -m domainbed.scripts.sweep ${command} --data_dir=${data_dir} \
    --output_dir=${output_dir}  --command_launcher gpu_0 --algorithms ${algorithms}  \
    --single_domain_gen  --datasets ${datasets}  --n_hparams 1 --n_trials 1 --continue_checkpoint ${checkpoint_URL} \
    --hparams """{\"batch_size\":64,\"lr\":0.001,\"kernel_size\":0.0,\"consistency_loss_w\":5.0,\"resnet_dropout\":0.0,\"val_augmentation\":true,\"alpha_min\":0.0,\"alpha_max\":1.0,\"weight_decay\":0.0,\"custom_train_val\":true,\"custom_train\":0,\"custom_val\":1,\"resnet18\":false,\"empty_fc\":true,\"unfreeze_train_bn\":true}"""
done

# nohup bash bash_scripts/ERM-ViT/ImageNet/ResNet/RandConv_Second/Lam_5_SGD_Val_Aug_BN_Only_Clipped.sh > OUT/SGD_RandConv_Second/Lam_5_SGD_Val_Aug_BN_Only_Clipped.out 2>&1 &