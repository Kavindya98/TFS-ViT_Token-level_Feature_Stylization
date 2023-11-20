for algo in ViTBase 
do
    python -u evaluvation.py --data_dir /media/SSD2/kavindya/Model/backgrounds_challenge --dataset ImageNet_9 \
    --output_dir /home/kavindya/data/Model/TFS-ViT_Token-level_Feature_Stylization/Results/ImageNet_9/ERM/${algo} --algorithm ${algo} --saved_model_evaluvator
done

# ResNet50 DeiTBase