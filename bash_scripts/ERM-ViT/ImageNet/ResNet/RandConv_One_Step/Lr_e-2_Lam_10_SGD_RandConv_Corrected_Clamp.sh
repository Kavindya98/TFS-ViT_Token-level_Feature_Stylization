algorithms=RandConv_CNN
datasets=ImageNet # PACS, VLCS, OfficeHome, TerraIncognita, DomainNet, ImageNet_9 
data_dir=/media/SSD2/Dataset 
output_dir=./Results/${datasets}/Fullset/SGD_ReNet/RandConv_One_Step/Lr_e-2_Lam_10_Val_Aug_Full_Model_Corrected_Clamp
for command in delete_incomplete launch
do
    python -u -m domainbed.scripts.sweep ${command} --data_dir=${data_dir} \
    --output_dir=${output_dir}  --command_launcher gpu_1 --algorithms ${algorithms}  \
    --single_domain_gen  --datasets ${datasets}  --n_hparams 1 --n_trials 1  \
    --hparams """{\"batch_size\":64,\"lr\":0.001,\"kernel_size\":0.0,\"resnet_dropout\":0.0,\"val_augmentation\":true,\"alpha_min\":0.0,\"alpha_max\":1.0,\"weight_decay\":0.0,\"custom_train_val\":true,\"custom_train\":0,\"custom_val\":1,\"resnet18\":false,\"empty_fc\":false,\"fixed_featurizer\":false}"""
done

# nohup bash bash_scripts/ERM-ViT/ImageNet/ResNet/RandConv_One_Step/Lr_e-2_Lam_10_SGD_RandConv_Corrected_Clamp.sh > OUT/SGD_RandConv_One_Step/Lr_e-2_Lam_10_SGD_RandConv_Corrected_Clamp.out 2>&1 &