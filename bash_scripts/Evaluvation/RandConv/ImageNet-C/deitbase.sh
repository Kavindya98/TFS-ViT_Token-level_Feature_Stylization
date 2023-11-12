for folder in RandConv_ViT/DeiTBase
do
    for file in t1_s0 
    do 
        python -u evaluvation.py --data_dir /media/SSD2/Dataset --dataset ImageNet_C \
        --output_dir /home/kavindya/data/Model/TFS-ViT_Token-level_Feature_Stylization/Results/ImageNet-C/Subset_ImageNet/${folder}/${file} \
        --saved_model /home/kavindya/data/Model/TFS-ViT_Token-level_Feature_Stylization/Results/ImageNet/New/${folder}/${file}/IID_best.pkl

        python -u imagnet_c_eval.py --input_dir /home/kavindya/data/Model/TFS-ViT_Token-level_Feature_Stylization/Results/ImageNet-C/Subset_ImageNet/${folder}/${file}
    done
done

# python -u evaluvation.py --data_dir /media/SSD2/Dataset --dataset ImageNet_C \
#     --output_dir /home/kavindya/data/Model/TFS-ViT_Token-level_Feature_Stylization/Results/ImageNet-C/RandConv_CNN_2/t1_s1 \
#      --saved_model /home/kavindya/data/Model/TFS-ViT_Token-level_Feature_Stylization/Results/ImageNet/RandConv_CNN_2/t1_s1/model.pkl

# for file in t1_s0 t1_s1 
#     do
#     for model_path in model.pkl IID_best.pkl
#         do 
#             python -u evaluvation.py --data_dir /media/SSD2/Dataset --dataset ImageNet_C \
#             --output_dir /home/kavindya/data/Model/TFS-ViT_Token-level_Feature_Stylization/Results/ImageNet-C/RandConv_CNN_C/${file}_${model_path}_ \
#             --saved_model /home/kavindya/data/Model/TFS-ViT_Token-level_Feature_Stylization/Results/ImageNet/RandConv_CNN_C/${file}/${model_path}
#         done
#     done

# for file in t1_s0 t1_s1 t1_s2
#     do 
#         python -u evaluvation.py --data_dir /media/SSD2/Dataset --dataset ImageNet_C \
#         --output_dir /home/kavindya/data/Model/TFS-ViT_Token-level_Feature_Stylization/Results/ImageNet-C/RandConv_ViT/DeiTBase/${file}_IID_best \
#         --saved_model /home/kavindya/data/Model/TFS-ViT_Token-level_Feature_Stylization/Results/ImageNet/RandConv_ViT/DeiTBase/${file}/IID_best.pkl
#     done