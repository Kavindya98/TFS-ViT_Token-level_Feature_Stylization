for algo in ViTBase
do
    python -u evaluvation.py --data_dir /media/SSD2/Dataset --dataset ImageNet_C \
    --output_dir /home/kavindya/data/Model/TFS-ViT_Token-level_Feature_Stylization/Results/ImageNet-C/Subset_ImageNet/ERM2/${algo} --algorithm ${algo} --saved_model_evaluvator

    python -u imagnet_c_eval.py --input_dir /home/kavindya/data/Model/TFS-ViT_Token-level_Feature_Stylization/Results/ImageNet-C/Subset_ImageNet/ERM2/${algo} 
done
for algo in ViTBase 
do
    python -u evaluvation.py --data_dir /media/SSD2/Dataset --dataset ImageNet_9 \
    --output_dir /home/kavindya/data/Model/TFS-ViT_Token-level_Feature_Stylization/Results/ImageNet_9/ERM_VAL/${algo} --algorithm ${algo} --saved_model_evaluvator
done

# ResNet50 DeiTBase