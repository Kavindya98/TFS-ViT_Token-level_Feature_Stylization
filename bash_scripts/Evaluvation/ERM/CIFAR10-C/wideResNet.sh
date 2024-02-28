python -u evaluvation.py --data_dir /media/SSD2/Dataset --dataset CIFAR10C \
--output_dir /home/kavindya/data/Model/TFS-ViT_Token-level_Feature_Stylization/Results/CIFAR10C/AugMix2_my_eval_e_100_with_nesterov  \
--saved_model /home/kavindya/data/Model/TFS-ViT_Token-level_Feature_Stylization/Results/CIFAR10/AugMix2_e_100_with_nesterov/t1_s2/IID_best.pkl \
--device cuda:0

# nohup bash bash_scripts/Evaluvation/ERM/CIFAR10-C/wideResNet.sh > OUT/CIFAR10C/AugMix2_my_eval_e_100_with_nesterov.out 2>&1 &

# nohup bash bash_scripts/Evaluvation/ERM/ImageNet/rc_resnet50.sh > OUT/SGD_ImageNet_C/RandConv_One_Step/RandConv_Val_Aug_Full_Model_Clipped.out 2>&1 &