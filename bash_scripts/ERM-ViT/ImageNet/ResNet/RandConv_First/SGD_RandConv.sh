algorithms=RandConv_CNN
datasets=ImageNet # PACS, VLCS, OfficeHome, TerraIncognita, DomainNet, ImageNet_9 
data_dir=/media/SSD2/Dataset 
output_dir=./Results/${datasets}/Fullset/SGD_ReNet/RandConv/Fixed_Feature_Clipped_No_Invariant
for command in delete_incomplete launch
do
    python -u -m domainbed.scripts.sweep ${command} --data_dir=${data_dir} \
    --output_dir=${output_dir}  --command_launcher gpu_2 --algorithms ${algorithms}  \
    --single_domain_gen  --datasets ${datasets}  --n_hparams 1 --n_trials 1  \
    --hparams """{\"batch_size\":64,\"lr\":0.001,\"kernel_size\":0.0,\"identity_prob\":0.5,\"resnet_dropout\":0.0,\"val_augmentation\":false,\"invariant_loss\":false,\"consistency_loss_w\":0.0,\"alpha_min\":0.0,\"alpha_max\":1.0,\"weight_decay\":0.0,\"custom_train_val\":true,\"custom_train\":0,\"custom_val\":1,\"resnet18\":false,\"empty_fc\":true,\"fixed_featurizer\":true}"""
done

# nohup bash bash_scripts/ERM-ViT/ImageNet/ResNet/SGD_RandConv.sh > OUT/SGD/Fixed_Feature_Clipped_No_Invariant.out 2>&1 &