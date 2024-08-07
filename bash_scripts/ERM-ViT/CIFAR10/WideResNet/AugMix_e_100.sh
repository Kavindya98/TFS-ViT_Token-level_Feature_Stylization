algorithms=AugMix_CNN
datasets=CIFAR10 # PACS, VLCS, OfficeHome, TerraIncognita, DomainNet, ImageNet_9 
data_dir=/media/SSD2/Dataset 
output_dir=./Results/${datasets}/AugMix_e_100_with_nesterov
for command in delete_incomplete launch
do
    python -u -m domainbed.scripts.sweep ${command} --data_dir=${data_dir} \
    --output_dir=${output_dir}  --command_launcher gpu_3 --algorithms ${algorithms}  \
    --single_domain_gen  --datasets ${datasets}  --n_hparams 1 --n_trials 1  \
    --hparams """{\"batch_size\":128,\"lr\":0.1,\"digits\":false,\"scheduler\":true,\"nesterov\":true,\"consistency_loss_w\":12.0,\"weight_decay\":0.0005,\"custom_train_val\":true,\"custom_train\":0,\"custom_val\":1,\"empty_fc\":true,\"fixed_featurizer\":false}"""
done

# nohup bash bash_scripts/ERM-ViT/CIFAR10/WideResNet/AugMix_e_100.sh > OUT/CIFAR10/WideResNet/AugMix_e_100_with_nesterov.out 2>&1 &