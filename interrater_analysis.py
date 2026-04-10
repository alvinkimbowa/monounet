import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from evaluate_in_orig_space import (
    DiceMetric,
    HausdorffDistanceMetric,
    SurfaceDistanceMetric,
    binarize_mask,
    compute_hd95_in_mm,
    compute_masd_in_mm,
    compute_monai_metrics,
    load_pixels_per_mm_map,
)
from preprocess_interrater_data import DEFAULT_OUTPUT_ROOT


RESOLUTION_CSV = Path("/home/ultrai/UltrAi/knee_us_segmentation/data/pixel_physical_resolution.csv")
RAW_DATASET_TO_ID = {
    "ACLMRI_Michel": 72,
    "Nike_Brian": 72,
    "Nike_Jagger": 72,
    "Nike_Michel": 72,
    "Nike_Olivia": 72,
    "Nike_XC_Jamil": 72,
    "Nike_Arjun_ge_auto": 72,
    "Nike_Arjun_ge_std": 72,
    "CaseControl_ACL": 73,
    "CaseControl_Control": 73,
    "Tufts_3m_Sunny": 73,
    "Tufts_6m_Jamil": 73,
    "Tufts_preop_Harkey": 73,
    "Nike_Arjun_clarius_auto": 70,
    "Nike_Arjun_clarius_std": 70,
}
DATASET_ID_TO_NAME = {
    70: "Dataset070_Clarius_L15",
    72: "Dataset072_GE_LQP9",
    73: "Dataset073_GE_LE",
}
ANNOTATORS = ("Will", "Ibrahim")


def get_pixels_per_mm(expert_mask_path, px_per_mm_map):
    raw_dataset_name = Path(expert_mask_path).parents[1].name
    dataset_id = RAW_DATASET_TO_ID.get(raw_dataset_name)
    if dataset_id is None:
        raise KeyError(f"Missing dataset-id mapping for raw dataset {raw_dataset_name}")
    px_per_mm = px_per_mm_map.get(dataset_id)
    if px_per_mm is None:
        raise KeyError(f"Missing pixels-per-mm mapping for dataset id {dataset_id}")
    return px_per_mm


def get_dataset_name(expert_mask_path):
    raw_dataset_name = Path(expert_mask_path).parents[1].name
    dataset_id = RAW_DATASET_TO_ID.get(raw_dataset_name)
    if dataset_id is None:
        raise KeyError(f"Missing dataset-id mapping for raw dataset {raw_dataset_name}")
    dataset_name = DATASET_ID_TO_NAME.get(dataset_id)
    if dataset_name is None:
        raise KeyError(f"Missing dataset-name mapping for dataset id {dataset_id}")
    return dataset_name


def load_manifest(output_root):
    manifest_path = output_root / "interrater_manifest.csv"
    manifest_df = pd.read_csv(manifest_path)
    return manifest_path, manifest_df


def compute_rater_results(output_root, resolution_csv=RESOLUTION_CSV):
    manifest_path, manifest_df = load_manifest(output_root)
    px_per_mm_map = load_pixels_per_mm_map(resolution_csv)
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    hd95_metric = HausdorffDistanceMetric(include_background=False, reduction="mean", percentile=95)
    masd_metric = SurfaceDistanceMetric(include_background=False, reduction="mean")

    result_paths = []
    for annotator in ANNOTATORS:
        annotator_df = manifest_df[manifest_df["annotator"] == annotator].copy()
        result_rows = []
        for _, row in tqdm(
            annotator_df.iterrows(),
            total=len(annotator_df),
            desc=f"Analyzing {annotator}",
            leave=False,
        ):
            pred_mask = binarize_mask(np.array(Image.open(row["export_path"]).convert("L")))
            gt_mask = binarize_mask(np.array(Image.open(row["expert_mask_path"]).convert("L")))

            dice, hd95_px, masd_px = compute_monai_metrics(
                pred_mask,
                gt_mask,
                dice_metric,
                hd95_metric,
                masd_metric,
            )

            px_per_mm = get_pixels_per_mm(row["expert_mask_path"], px_per_mm_map)
            result_rows.append(
                {
                    "image_id": row["label_style_image_id"],
                    "dataset": get_dataset_name(row["expert_mask_path"]),
                    "dice": round(dice * 100, 2),
                    "hd95": round(compute_hd95_in_mm(hd95_px, px_per_mm), 2),
                    "masd": round(compute_masd_in_mm(masd_px, px_per_mm), 2),
                }
            )

        result_df = pd.DataFrame(result_rows).sort_values(["dataset", "image_id"])
        annotator_root = output_root / annotator
        annotator_root.mkdir(parents=True, exist_ok=True)
        for dataset_name, dataset_df in result_df.groupby("dataset", sort=False):
            dataset_root = annotator_root / dataset_name
            dataset_root.mkdir(parents=True, exist_ok=True)
            result_path = dataset_root / f"results_{dataset_name}.csv"
            dataset_df[["image_id", "dice", "hd95", "masd"]].to_csv(result_path, index=False)
            result_paths.append(result_path)

    return manifest_path, result_paths


def parse_args():
    parser = argparse.ArgumentParser(description="Compute interrater metrics against original labels.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory containing interrater masks and manifest, and where results CSVs will be written.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    manifest_path, result_paths = compute_rater_results(args.output_root)
    print(f"Manifest: {manifest_path}")
    for result_path in result_paths:
        print(f"Results: {result_path}")


if __name__ == "__main__":
    main()
