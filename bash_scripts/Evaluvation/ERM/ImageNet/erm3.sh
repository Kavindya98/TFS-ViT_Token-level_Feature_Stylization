for algo in T2T14
do
    python -u evaluvation.py --data_dir /media/SSD2/Dataset --dataset ImageNet_C \
    --output_dir /home/kavindya/data/Model/TFS-ViT_Token-level_Feature_Stylization/Results/ImageNet-C_ALL/T2T14 --algorithm ${algo} \
    --device cuda:0 --saved_model_evaluvator

    #python -u imagnet_c_eval.py --input_dir /home/kavindya/data/Model/TFS-ViT_Token-level_Feature_Stylization/Results/ImageNet-C/Fullset_ImageNet/ERM_Normalized_0.5/${algo} 
done

# nohup bash bash_scripts/Evaluvation/ERM/ImageNet/erm3.sh > OUT/ImageNet-C_ALL/T2T14/ERM.out 2>&1 &