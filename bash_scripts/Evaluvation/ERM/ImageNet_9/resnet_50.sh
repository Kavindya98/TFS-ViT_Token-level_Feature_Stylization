for algo in ResNet50_Long
do
    python -u evaluvation.py --data_dir /media/SSD2/Dataset --dataset ImageNet_val \
    --output_dir /home/kavindya/data/Model/TFS-ViT_Token-level_Feature_Stylization/Results/ImageNet_9/RandConv/${algo} \
    --saved_model /home/kavindya/data/Model/TFS-ViT_Token-level_Feature_Stylization/Results/ImageNet/Fullset/Long_Epoch_Clipped/RandConv/t1_s2/IID_best.pkl
done

# ResNet50 DeiTBase
# python -u evaluvation.py --data_dir /media/SSD2/kavindya/Model/backgrounds_challenge --dataset ImageNet_9 \
#     --output_dir /home/kavindya/data/Model/TFS-ViT_Token-level_Feature_Stylization/Results/ImageNet_9/ERM/${algo} --algorithm ${algo} --saved_model_evaluvator