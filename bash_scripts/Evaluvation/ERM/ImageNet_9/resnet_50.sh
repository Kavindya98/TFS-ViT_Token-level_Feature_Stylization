for algo in DeiTBase ResNet50 ResNet18
do
    python -u evaluvation.py --data_dir /media/SSD2/Dataset --dataset ImageNet_9 \
    --output_dir /home/kavindya/data/Model/TFS-ViT_Token-level_Feature_Stylization/Results/ImageNet_9/ERM/${algo} --algorithm ${algo} --saved_model_evaluvator
done

# ResNet50 DeiTBase