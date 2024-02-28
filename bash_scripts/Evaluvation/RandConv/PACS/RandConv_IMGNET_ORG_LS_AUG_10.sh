
for data in PACS  
do
    python -u evaluvation.py --data_dir /media/SSD2/Dataset --dataset ${data} \
    --output_dir /home/kavindya/data/Model/TFS-ViT_Token-level_Feature_Stylization/Results/PACS/Evaluation/Lr_e-3_Lam_5_SGD_RandConv_Corrected_Clamp_Loss_Aug_Train_BN_0 \
    --saved_model /home/kavindya/data/Model/TFS-ViT_Token-level_Feature_Stylization/Results/PACS/RandConv_One_Step/Lr_e-3_Lam_5_SGD_RandConv_Corrected_Clamp_Loss_Aug_Train_BN/t1234_s0/IID_best.pkl \
    --device cuda:0

    #python -u imagnet_c_eval.py --input_dir /home/kavindya/data/Model/TFS-ViT_Token-level_Feature_Stylization/Results/ImageNet-C/Fullset_ImageNet/ERM_Normalized_0.5/${algo} 
done


# nohup bash bash_scripts/Evaluvation/RandConv/PACS/RandConv_IMGNET_ORG_LS_AUG_10.sh > OUT/PACS/Evaluation/Lr_e-3_Lam_5_SGD_RandConv_Corrected_Clamp_Loss_Aug_Train_BN_0.out 2>&1 &