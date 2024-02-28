algorithms=ERM
datasets=ImageNet # PACS, VLCS, OfficeHome, TerraIncognita, DomainNet, ImageNet_9 
data_dir=/media/SSD2/Dataset 
output_dir=./Results/${datasets}/Fullset/SGD_ReNet/ERM/Fixed_Feature
for command in delete_incomplete launch
do
    python -u -m domainbed.scripts.sweep ${command} --data_dir=${data_dir} \
    --output_dir=${output_dir}  --command_launcher gpu_0 --algorithms ${algorithms}  \
    --single_domain_gen  --datasets ${datasets}  --n_hparams 1 --n_trials 1  \
    --hparams """{\"batch_size\":64,\"lr\":0.001,\"resnet_dropout\":0.0,\"weight_decay\":0.0,\"custom_train_val\":true,\"custom_train\":0,\"custom_val\":1,\"resnet18\":false,\"empty_fc\":true,\"fixed_featurizer\":true}"""
done

# nohup bash bash_scripts/ERM-ViT/ImageNet/ResNet/SGD_ERM.sh > OUT/SGD/ERM_Fixed_Feature.out 2>&1 &