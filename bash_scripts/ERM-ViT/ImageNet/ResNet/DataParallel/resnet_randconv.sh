python -m domainbed.scripts.train --algorithm RandConv_CNN --continue_checkpoint ' ' \
 --data_dir /media/SSD2/Dataset --dataset ImageNet --holdout_fraction 0.2 \
 --hparams '{"batch_size":64,"lr":0.0001,"kernel_size":0.0,"resnet_dropout":0.0,"val_augmentation":true,"alpha_min":0.0,"alpha_max":1.0,"weight_decay":0.0,"custom_train_val":true,"custom_train":0,"custom_val":1,"resnet18":false,"empty_fc":false,"fixed_featurizer":false}' \
 --hparams_seed 0 --output_dir ./Results/ImageNet/Fullset/SGD_ReNet/RandConv_One_Step/Lr_e-3_Lam_10_SGD_RandConv_Corrected_Clamp_Parallel \
 --seed 1637210862 --task domain_generalization --test_envs 1 --trial_seed 2

 # nohup bash bash_scripts/ERM-ViT/ImageNet/ResNet/DataParallel/resnet_randconv.sh > OUT/DataParallel/Lr_e-3_Lam_10_SGD_RandConv_Corrected_Clamp_Parallel.out 2>&1 &