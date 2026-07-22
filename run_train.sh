#!/bin/bash

nnUNet_raw="../knee_us_segmentation/data/nnUNet_raw"
nnUNet_preprocessed="../knee_us_segmentation/data/nnUNet_preprocessed"
MODELS_DIR="models"

export nnUNet_raw=$nnUNet_raw
export nnUNet_preprocessed=$nnUNet_preprocessed
export NO_ALBUMENTATIONS_UPDATE=1

train=1
eval=1
analyze=1
# dataset_name="Dataset072_GE_LQP9"
dataset_name="Dataset073_GE_LE"
# dataset_name="Dataset070_Clarius_L15"
lr=0.0001
epochs=1000
b=8

resume_ckpt="auto"   # "auto" uses <MODELS_DIR>/<arch_name>/<dataset>/fold_<fold>/model_latest.pth

gpu=0
export CUDA_VISIBLE_DEVICES=$gpu

# Evaluation settings
save_preds=true
overlay=false
data_augmentation=true
largest_component=true
test_datasets=("Dataset072_GE_LQP9" "Dataset073_GE_LE" "Dataset070_Clarius_L15" "Dataset079_KneeUS_Ilker")
ckpt="model_best.pth"

# Architecture list - comment/uncomment to select which models to use
# For train/eval: uncomment only ONE architecture
# For analyze: uncomment ALL architectures you want to analyze
all_archs=(
    # "UNext"
    # "TinyUNet"
    # "UNetBaseline"
    # "XTinynnUNet"
    # "MonoUNetBase"
    # "MonoUNetE1"
    # "MonoUNetE123"
    # "MonoUNetE123V2"
    "MonoUNetE123V2Gated"
)

# Get the first uncommented architecture for train/eval
arch="${all_archs[0]}"
if [[ $arch =~ ^[[:space:]]*# ]]; then
    echo "Error: First architecture in all_archs is commented. Please uncomment at least one architecture."
    exit 1
fi

if [[ $arch == "UNet" ]]; then
    min_lr=1e-5
    loss="UNetLoss"
    input_h=256
    input_w=256
    optimizer="SGD"
    momentum=0.99
    scheduler="ConstantLR"
    weight_decay=1e-4
    input_channels=1
    deep_supervision=False
    num_classes=2
elif [[ $arch == "TinyUNet" ]]; then
    min_lr=1e-6
    loss="TinyUNetLoss"
    input_h=256
    input_w=256
    optimizer="Adam"
    scheduler="CosineAnnealingLR"
    weight_decay=1e-4
    deep_supervision=False
    input_channels=3
    num_classes=1
elif [[ $arch == XTiny* || $arch == MonoUNet* || $arch == UNetBaseline ]]; then
    lr=0.01
    weight_decay=0.01
    min_lr=1e-5
    loss="TinyUNetLoss"
    input_h=256
    input_w=256
    deep_supervision=False
    optimizer="AdamW"
    scheduler="PolyLR"
    input_channels=1
    num_classes=1
else
    min_lr=1e-5
    loss="BCEDiceLoss"
    input_w=256
    input_h=256
    optimizer="Adam"
    scheduler="CosineAnnealingLR"
    weight_decay=1e-4
    deep_supervision=False
    input_channels=3
    num_classes=1
fi

for fold in {0..2}; do
    echo "fold: $fold"
    echo "train: $train"
    echo "eval: $eval"
    echo "analyze: $analyze"
    echo "dataset_name: $dataset_name"
    echo "arch: $arch"
    echo "exp_name: $exp_name"
    echo "lr: $lr"
    echo "epochs: $epochs"
    echo "input_w: $input_w"
    echo "input_h: $input_h"
    echo "b: $b"
    echo "input_channels: $input_channels"
    echo "deep_supervision: $deep_supervision"
    echo "data_augmentation: $data_augmentation"
    echo "largest_component: $largest_component"
    echo "num_classes: $num_classes"

    if [[ $train -eq 1 ]]; then
        arch_name="$arch"
        if [[ $deep_supervision == True || $deep_supervision == true ]]; then
            arch_name="${arch_name}DS"
        fi
        if [[ $data_augmentation == True || $data_augmentation == true ]]; then
            arch_name="${arch_name}DA"
        fi

        if [[ "$resume_ckpt" == "auto" || -z "$resume_ckpt" ]]; then
            # Search for either model_latest.pth (for resuming) or model_final.pth (if training completed)
            ckpt_dir="${MODELS_DIR}/${arch_name}/${dataset_name}/fold_${fold}"
            if [[ -f "${ckpt_dir}/model_latest.pth" ]]; then
                resume_ckpt="${ckpt_dir}/model_latest.pth"
            elif [[ -f "${ckpt_dir}/model_final.pth" ]]; then
                resume_ckpt="${ckpt_dir}/model_final.pth"
            else
                resume_ckpt="${ckpt_dir}/model_latest.pth"  # Default path, will be created if doesn't exist
            fi
        fi

        python train.py \
            --models_dir "$MODELS_DIR" \
            --dataset $dataset_name \
            --arch $arch \
            --lr $lr \
            --epochs $epochs \
            --input_w $input_w \
            --input_h $input_h \
            --b $b \
            --fold $fold \
            --min_lr $min_lr \
            --loss $loss \
            --optimizer $optimizer \
            --scheduler $scheduler \
            --weight_decay $weight_decay \
            --input_channels $input_channels \
            --deep_supervision $deep_supervision \
            --data_augmentation $data_augmentation \
            --num_classes $num_classes \
            --resume "$resume_ckpt"
    fi

    if [[ $eval -eq 1 ]]; then
        for test_dataset in ${test_datasets[@]}; do
            echo "Evaluating $test_dataset"
            if [[ $test_dataset == "Dataset078_KneeUS_OtherDevices" || $test_dataset == "Dataset079_KneeUS_Ilker" ]]; then
                test_split="Ts"
            else
                test_split="Tr"
            fi
            python val.py \
            --models_dir "$MODELS_DIR" \
            --name $arch \
            --train_dataset $dataset_name \
            --train_fold $fold \
            --test_dataset $test_dataset \
            --test_split $test_split \
            --save_preds $save_preds \
            --ckpt $ckpt \
            --deep_supervision $deep_supervision \
            --data_augmentation $data_augmentation \
            --overlay $overlay \
            --largest_component $largest_component
        done
    fi
done

if [[ $analyze -eq 1 ]]; then
    # Collect all uncommented architectures from all_archs array
    archs_to_analyze=()
    for a in "${all_archs[@]}"; do
        if [[ ! $a =~ ^[[:space:]]*# ]]; then
            archs_to_analyze+=("$a")
        fi
    done
    
    if [[ ${#archs_to_analyze[@]} -eq 0 ]]; then
        echo "Error: No architectures selected for analysis. Please uncomment at least one architecture in all_archs array."
        exit 1
    fi
    
    # Analyze all uncommented architectures
    for current_arch in "${archs_to_analyze[@]}"; do
        echo ""
        echo "============================================================"
        echo "Analyzing: $current_arch"
        echo "============================================================"
        
        # Determine settings based on architecture (same logic as main script)
        if [[ $current_arch == "TinyUNet" ]]; then
            analyze_input_h=256
            analyze_input_w=256
            analyze_deep_supervision=False
        elif [[ $current_arch == XTiny* || $current_arch == MonoUNet* ]]; then
            analyze_input_h=256
            analyze_input_w=256
            analyze_deep_supervision=False
        else
            analyze_input_w=512
            analyze_input_h=512
            analyze_deep_supervision=False
        fi
        
        analyze_args="--arch $current_arch --input_channels $input_channels --input_h $analyze_input_h --input_w $analyze_input_w --gpu $gpu --deep_supervision $analyze_deep_supervision"
        
        # Save analysis to the train-dataset directory if it exists
        if [[ $data_augmentation == true ]]; then
            current_arch="$current_arch"DA
        fi

        model_dir="$MODELS_DIR/$current_arch/$dataset_name"
        echo "model_dir: $model_dir"
        if [[ -d "$model_dir" ]]; then
            analyze_args="$analyze_args --save $model_dir/model_analysis.json"
        fi
        
        python analyze_model.py $analyze_args
        
        echo "✓ Completed analysis for $current_arch"
    done
    
    echo ""
    echo "============================================================"
    echo "All models analyzed!"
    echo "============================================================"
fi
