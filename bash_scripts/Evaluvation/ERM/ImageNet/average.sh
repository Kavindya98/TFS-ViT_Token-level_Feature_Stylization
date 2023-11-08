for folder in ResNet50 DeiTBase ViTBase
    do
        python -u imagnet_c_eval.py --input_dir /home/kavindya/data/Model/TFS-ViT_Token-level_Feature_Stylization/Results/ImageNet-C/ERM2/${folder} 

    done

