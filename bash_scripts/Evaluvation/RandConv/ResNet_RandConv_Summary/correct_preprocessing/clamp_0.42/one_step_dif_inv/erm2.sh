for algo in ResNet50
do
    python -u evaluvation.py --data_dir /media/SSD2/Dataset --dataset ImageNet_val \
    --output_dir /home/kavindya/data/Model/TFS-ViT_Token-level_Feature_Stylization/Results/ImageNet-val/SGD_ReNet/ERM2 --algorithm ${algo} \
    --device cuda:0 --saved_model_evaluvator

    #python -u imagnet_c_eval.py --input_dir /home/kavindya/data/Model/TFS-ViT_Token-level_Feature_Stylization/Results/ImageNet-C/Fullset_ImageNet/ERM_Normalized_0.5/${algo} 
done

# nohup bash bash_scripts/Evaluvation/RandConv/ResNet_RandConv_Summary/correct_preprocessing/clamp_0.42/one_step_dif_inv/erm2.sh > OUT/ResNet_RandConv_Summary/correct_preprocessing/clamp_0.42/one_step_dif_inv/ERM2.out 2>&1 &