algorithms=AugMix_ViT
datasets=PACS # PACS, VLCS, OfficeHome, TerraIncognita, DomainNet, ImageNet_9 
backbone=ViTBase # DeitSmall, T2T14# DeitSmall, T2T14
data_dir=/media/SSD2/Dataset

for command in delete_incomplete launch
    do 
        output_dir=./Results/${datasets}/${algorithms}/${backbone}_test
        python -u -m domainbed.scripts.sweep ${command} --data_dir=${data_dir} \
        --output_dir=${output_dir}_test  --command_launcher gpu_0 --algorithms ${algorithms}  \
        --single_domain_gen  --datasets ${datasets}  --n_hparams 1 --n_trials 1  \
        --hparams """{\"backbone\":\"${backbone}\",\"consistency_loss_w\":12.0,\"normalization\":false,\"batch_size\":64,\"lr\":5e-05 ,\"resnet_dropout\":0.0,\"val_augmentation\":true,\"weight_decay\":0.0,\"fixed_featurizer\":false,\"empty_head\":true}"""
    done

# nohup bash bash_scripts/ERM-ViT/PACS/run_sweep_AugMix_DeiT_loss_aug.sh > OUT/PACS/AugMix_ViT/ViTBase_AugMix_test.out 2>&1 &
