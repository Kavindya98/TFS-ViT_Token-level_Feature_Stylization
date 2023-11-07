for folder in RandConv_ViT/DeiTBase RandConv_ViT/ViTBase
do
    for file in t1_s0 t1_s1 t1_s2
    do 
        python -u evaluvation.py --data_dir /media/SSD2/Dataset --dataset ImageNet_C \
        --output_dir /home/kavindya/data/Model/TFS-ViT_Token-level_Feature_Stylization/Results/ImageNet-C/${folder}/${file} \
        --saved_model /home/kavindya/data/Model/TFS-ViT_Token-level_Feature_Stylization/Results/ImageNet/${folder}/${file}/model.pkl
    done
done

# python -u evaluvation.py --data_dir /media/SSD2/Dataset --dataset ImageNet_C \
#     --output_dir /home/kavindya/data/Model/TFS-ViT_Token-level_Feature_Stylization/Results/ImageNet-C/RandConv_CNN_2/t1_s1 \
#      --saved_model /home/kavindya/data/Model/TFS-ViT_Token-level_Feature_Stylization/Results/ImageNet/RandConv_CNN_2/t1_s1/model.pkl