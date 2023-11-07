for algo in ResNet50 DeiTBase ViTBase
do
    python -u evaluvation.py --data_dir /media/SSD2/Dataset --dataset ImageNet_C \
    --output_dir /home/kavindya/data/Model/TFS-ViT_Token-level_Feature_Stylization/Results/ImageNet-C/ERM2/${algo} --algorithm ${algo} --saved_model_evaluvator
done
 