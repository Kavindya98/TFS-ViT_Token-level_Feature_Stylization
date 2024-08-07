
for data in ImageNet_val  
do
    python -u evaluvation.py --data_dir /media/SSD2/Dataset --dataset ${data} \
    --output_dir /home/kavindya/data/Model/TFS-ViT_Token-level_Feature_Stylization/Results/ImageNet-C/ResNet_RandConv_Summary/clamp_0.5/one_step/RandConv_One_Step_Lam_5_Val_Aug_Full_Model_Clipped \
    --saved_model /home/kavindya/data/Model/TFS-ViT_Token-level_Feature_Stylization/Results/ImageNet/Fullset/SGD_ReNet/RandConv_One_Step/Lam_5_Val_Aug_Full_Model_Clipped/t1_s2/IID_best.pkl \
    --device cuda:0

    #python -u imagnet_c_eval.py --input_dir /home/kavindya/data/Model/TFS-ViT_Token-level_Feature_Stylization/Results/ImageNet-C/Fullset_ImageNet/ERM_Normalized_0.5/${algo} 
done


# nohup bash bash_scripts/Evaluvation/RandConv/ResNet_RandConv_Summary/correct_preprocessing/clamp_0.5/one_step/RandConv_One_Step_ImageNet_Val_Lam_5_Val_Aug_Full_Model_Clipped.sh > OUT/ResNet_RandConv_Summary/correct_preprocessing/clamp_0.5/one_step/RandConv_One_Step_ImageNet_Val_Lam_5_Val_Aug_Full_Model_Clipped.out 2>&1 &