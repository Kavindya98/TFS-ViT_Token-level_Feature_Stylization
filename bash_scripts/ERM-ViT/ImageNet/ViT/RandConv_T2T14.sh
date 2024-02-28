algorithms=RandConv_ViT
datasets=ImageNet # PACS, VLCS, OfficeHome, TerraIncognita, DomainNet, ImageNet_9 
backbone=T2T14 # DeitSmall, T2T14# DeitSmall, T2T14
data_dir=/media/SSD2/Dataset
#output_dir=./Results/${datasets}/ERM_VIT/DeitBase
for command in delete_incomplete launch
    do 
        output_dir=./Results/${datasets}/${algorithms}/${backbone}
        python -u -m domainbed.scripts.sweep ${command} --data_dir=${data_dir} \
        --output_dir=${output_dir}  --command_launcher multi_gpu --algorithms ${algorithms}  \
        --single_domain_gen  --datasets ${datasets}  --n_hparams 1 --n_trials 1  \
        --hparams """{\"backbone\":\"${backbone}\",\"kernel_size\":0.0,\"consistency_loss_w\":10.0,\"custom_train_val\":true,\"custom_train\":0,\"custom_val\":1,\"loss_aug\":true,\"alpha_min\":0.0,\"alpha_max\":1.0,\"batch_size\":64,\"lr\":5e-05 ,\"resnet_dropout\":0.0,\"val_augmentation\":true,\"weight_decay\":0.0,\"fixed_featurizer\":false,\"empty_head\":false}"""
    done