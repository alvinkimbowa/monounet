#!/usr/bin/env bash
set -euo pipefail

export nnUNet_raw="data/nnUNet_raw"
export nnUNet_results="data/nnUNet_results"
ORIG_DATASETS="data/raw_data/datasets"
RESOLUTION_CSV="data/pixel_physical_resolution.csv"
MODELS_DIR="models"

# models=(UNext Med_NCA CMUNeXt-S TinyUNet)
# models=(UNextDA Med_NCADA CMUNeXt-SDA TinyUNetDA)
# models=(UNetBaseline MonoUNetBase MonoUNetE1 MonoUNetE123 MonoUNetE123V2 MonoUNetE123V2Gated MonoUNetE123V2GatedDA)
models=(MonoUNetE123V2GatedDA)

# train_dataset_ids=(73 70 72)
train_dataset_ids=(73)
# test_dataset_ids=(72 73 70 79)
test_dataset_ids=(70)

DEBUG=false
SHUFFLE=false
RECOMPUTE_EXISTING=false
COMPUTE="performance"
# COMPUTE="outcomes"

for MODEL in "${models[@]}"; do
	for FOLD in {0..2}; do
		for TRAIN_DATASET_ID in "${train_dataset_ids[@]}"; do
			for TEST_DATASET_ID in "${test_dataset_ids[@]}"; do
				if [ "$TEST_DATASET_ID" == 79 ]; then
					SPLIT="Ts"
				else
					SPLIT="Tr"
				fi
				echo "Evaluating $MODEL: Fold $FOLD, Train $TRAIN_DATASET_ID, Test $TEST_DATASET_ID"
				ARGS=(
					--models_dir "$MODELS_DIR"
					--model "$MODEL"
					--fold "$FOLD"
					--train_dataset_id "$TRAIN_DATASET_ID"
					--test_dataset_id "$TEST_DATASET_ID"
					--split "$SPLIT"
					--orig_datasets "$ORIG_DATASETS"
					--resolution_csv "$RESOLUTION_CSV"
					--compute "$COMPUTE"
				)

				if [ "$DEBUG" = true ]; then
					ARGS+=(--debug)
				fi

				if [ "$SHUFFLE" = true ]; then
					ARGS+=(--shuffle)
				fi

				if [ "$RECOMPUTE_EXISTING" = true ]; then
					ARGS+=(--recompute_existing)
				fi

				python \
					evaluate_in_orig_space.py \
					"${ARGS[@]}"
			done
		done
	done
done
