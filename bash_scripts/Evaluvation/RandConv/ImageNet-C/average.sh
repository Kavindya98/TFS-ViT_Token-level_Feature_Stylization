# for folder in RandConv_ViT/DeiTBase RandConv_ViT/ViTBase
#     do
#     for file in t1_s0 t1_s1 t1_s2
#         do
#             python -u imagnet_c_eval.py --input_dir /home/kavindya/data/Model/TFS-ViT_Token-level_Feature_Stylization/Results/ImageNet-C/${folder}/${file} 
#         done

#     done

# for file in t1_s0 t1_s1 
#     do
#     for model_path in model.pkl IID_best.pkl
#         do 
#             python -u imagnet_c_eval.py --input_dir /home/kavindya/data/Model/TFS-ViT_Token-level_Feature_Stylization/Results/ImageNet-C/RandConv_CNN_C/${file}_${model_path}_
#         done
#     done

for file in RandConv_CNN RandConv_ViT/DeiTBase_2 RandConv_ViT/ViTBase_2
    do
    for model_path in model.pkl IID_best.pkl
        do 
            python -u imagnet_c_eval.py --input_dir /home/kavindya/data/Model/TFS-ViT_Token-level_Feature_Stylization/Results/ImageNet-C/${file}_${model_path}_
            
        done
    done