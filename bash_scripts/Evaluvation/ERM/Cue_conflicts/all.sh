for algo in DeitSmall T2T14 # ResNet50 ResNet18 DeiTBase ViTBase
do
    for data in Cue_conflicts ImageNet_V2 ImageNet_9 ImageNet_C
    do
    python -u evaluvation.py --data_dir /media/SSD2/Dataset --dataset ${data} \
    --output_dir /home/kavindya/data/Model/TFS-ViT_Token-level_Feature_Stylization/Results/ERM_ViT/${data}/${algo} --algorithm ${algo} --saved_model_evaluvator
    done
done

# nohup bash bash_scripts/Evaluvation/ERM/Cue_conflicts/all.sh > OUT/ERM/ERM_ViT.out 2>&1 &