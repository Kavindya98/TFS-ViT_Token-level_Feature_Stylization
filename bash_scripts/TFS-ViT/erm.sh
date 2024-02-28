algorithms=ERM_ViT
datasets=PACS # PACS, VLCS, OfficeHome, TerraIncognita, DomainNet, ImageNet_9 
#backbone=ViTBase DeiTBase # DeitSmall, T2T14# DeitSmall, T2T14
data_dir=/media/SSD2/Dataset
#output_dir=./Results/${datasets}/ERM_VIT/DeitBase

for command in delete_incomplete launch
do
    for backbone in T2T14  
    do 
        output_dir=./Results/${datasets}/ERM/${backbone}_PACS_unsplit
        python -m domainbed.scripts.sweep ${command} --data_dir=${data_dir} \
        --output_dir=${output_dir}  --command_launcher multi_gpu --algorithms ${algorithms}  \
        --single_domain_gen  --datasets ${datasets}  --n_hparams 1 --n_trials 3  \
        --hparams """{\"backbone\":\"${backbone}\",\"batch_size\":64,\"lr\":5e-05 ,\"resnet_dropout\":0.0,\"weight_decay\":0.0,\"fixed_featurizer\":false,\"empty_head\":true}"""
    done
done

# nohup bash bash_scripts/TFS-ViT/erm.sh > OUT/PACS/ERM/T2T14_PACS_unsplit.out 2>&1 &
# \"custom_train_val\":true,\"custom_train\":0,\"custom_val\":1,