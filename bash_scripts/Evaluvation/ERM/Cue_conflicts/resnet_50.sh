for algo in ResNet50 ResNet18 DeiTBase ViTBase
do
    python -u evaluvation.py --data_dir /media/SSD2/Dataset --dataset Cue_conflicts \
    --output_dir /home/kavindya/data/Model/TFS-ViT_Token-level_Feature_Stylization/Results/Cue_conflicts/ERM/${algo} --algorithm ${algo} --saved_model_evaluvator
done

# ResNet50 DeiTBase