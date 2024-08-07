for algo in ResNet50 
do
    python -u evaluvation.py --data_dir /media/SSD2/Dataset --dataset ImageNet_C \
    --output_dir /home/kavindya/data/Model/TFS-ViT_Token-level_Feature_Stylization/Results/ImageNet-C/Fullset_ImageNet/RC_Normalized_0.5/${algo}_long --algorithm ${algo} \
    --saved_model /home/kavindya/data/Model/TFS-ViT_Token-level_Feature_Stylization/Results/ImageNet/Fullset/Long_Epoch_Clipped/RandConv/t1_s2/IID_best.pkl \
    --device cuda:0

    #python -u imagnet_c_eval.py --input_dir /home/kavindya/data/Model/TFS-ViT_Token-level_Feature_Stylization/Results/ImageNet-C/Fullset_ImageNet/ERM_Normalized_0.5/${algo} 
done
# for algo in ResNet50 
# do
#     python -u evaluvation.py --data_dir /media/SSD2/Dataset --dataset ImageNet_val \
#     --output_dir /home/kavindya/data/Model/TFS-ViT_Token-level_Feature_Stylization/Results/ImageNet_9/ERM_VAL/${algo} --algorithm ${algo} --saved_model_evaluvator
# done

# ResNet50 DeiTBase

# nohup bash bash_scripts/Evaluvation/ERM/ImageNet/rc_deit.sh > long_resnet_c.out 2>&1 &