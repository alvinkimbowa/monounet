#!/usr/bin/env python3

import argparse
import csv
import os
import random
from glob import glob
from os.path import join

import numpy as np
import pandas as pd
import torch
from PIL import Image
from matplotlib import pyplot as plt
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from skimage.morphology import skeletonize
from tqdm import tqdm


IGNORE_IDS = {
    "Nike_Brian___NIKE_S962_T4_R_CAR_3_1562",
    "Nike_Brian___NIKE_S975_T4_L_CAR_2_752",
    "Nike_XC_Jamil___NIKE_S129_T0_L_CAR_3_1119",
    "Nike_XC_Jamil___NIKE_S129_T0_L_CAR_1_895",
    "Nike_Brian___NIKE_S951_T4_R_CAR_3_1149",
    "20251103201014____frame_73_042",
    "20251103200953____frame_8_254",
    "20251103200953____frame_11_229",
    "20251103200953____frame_46_040",
    "20251103200953____frame_48_354",
    "20251103200953____frame_49_274",
    "20251103200953____frame_50_063",
    "20251103201014____frame_2_366",
    "20251103201014____frame_3_048",
    "20251103201014____frame_4_003",
    "20251103201014____frame_45_257",
    "20251103201014____frame_46_317",
    "20251103201014____frame_47_263",
    "20251103201014____frame_48_152",
    "20251103201014____frame_67_265",
    "20251103201014____frame_68_090",
    "20251103201014____frame_69_101",
    "20251103201014____frame_70_009",
    "20251103201014____frame_71_148",
    "20251103201014____frame_72_202",
    "20251103201014____frame_74_243",
    "20251103201014____frame_75_231",
    "20251103201014____frame_76_021",
    "20251103201043____frame_7_301",
    "20251103201043____frame_8_099",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", default="models")
    parser.add_argument("--model", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--train_dataset_id", type=int, required=True)
    parser.add_argument("--test_dataset_id", type=int, required=True)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument(
        "--orig_datasets",
        default="/home/ultrai/UltrAi/knee_us_segmentation/data/raw_data/datasets",
    )
    parser.add_argument(
        "--resolution_csv",
        default="/home/ultrai/UltrAi/knee_us_segmentation/data/pixel_physical_resolution.csv",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--recompute_existing", action="store_true")
    parser.add_argument("--compute", choices=["performance", "outcomes"], default="performance")
    return parser.parse_args()


def get_dataset_name(nnunet_raw, dataset_id):
    dataset_id = str(dataset_id).zfill(3)
    dataset_names = [
        dataset_name
        for dataset_name in os.listdir(nnunet_raw)
        if dataset_name.startswith(f"Dataset{dataset_id}") and os.path.isdir(join(nnunet_raw, dataset_name))
    ]
    assert len(dataset_names) == 1, f"Found {len(dataset_names)} datasets with id {dataset_id}, expected 1"
    return dataset_names[0]


def get_bounding_box(mask):
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return 0, 0, mask.shape[1], mask.shape[0]

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    crop_w = x_max - x_min + 1
    crop_h = y_max - y_min + 1
    return x_min, y_min, crop_w, crop_h


def binarize_mask(mask):
    return (mask > 0).astype(np.uint8)


def compute_dice(mask_a, mask_b):
    mask_a = binarize_mask(mask_a)
    mask_b = binarize_mask(mask_b)
    intersection = np.logical_and(mask_a, mask_b).sum()
    denom = mask_a.sum() + mask_b.sum()
    if denom == 0:
        return 1.0
    return (2.0 * intersection) / denom


def get_surface(mask):
    mask = binarize_mask(mask).astype(bool)
    if not mask.any():
        return mask
    eroded = ndimage.binary_erosion(mask)
    return np.logical_xor(mask, eroded)


def compute_masd(mask_a, mask_b):
    mask_a = binarize_mask(mask_a)
    mask_b = binarize_mask(mask_b)

    if not mask_a.any() and not mask_b.any():
        return 0.0
    if not mask_a.any() or not mask_b.any():
        return np.nan

    surf_a = get_surface(mask_a)
    surf_b = get_surface(mask_b)
    dt_a = ndimage.distance_transform_edt(~surf_a)
    dt_b = ndimage.distance_transform_edt(~surf_b)
    dists_a_to_b = dt_b[surf_a]
    dists_b_to_a = dt_a[surf_b]
    return float(np.concatenate([dists_a_to_b, dists_b_to_a]).mean())


def compute_hd95(mask_a, mask_b):
    mask_a = binarize_mask(mask_a)
    mask_b = binarize_mask(mask_b)

    if not mask_a.any() and not mask_b.any():
        return 0.0
    if not mask_a.any() or not mask_b.any():
        return np.nan

    surf_a = get_surface(mask_a)
    surf_b = get_surface(mask_b)
    dt_a = ndimage.distance_transform_edt(~surf_a)
    dt_b = ndimage.distance_transform_edt(~surf_b)
    dists_a_to_b = dt_b[surf_a]
    dists_b_to_a = dt_a[surf_b]
    return float(np.percentile(np.concatenate([dists_a_to_b, dists_b_to_a]), 95))


def compute_monai_metrics(mask_a, mask_b, dice_metric, hd95_metric, masd_metric):
    pred = torch.from_numpy(binarize_mask(mask_a)[None, None].astype(np.float32))
    target = torch.from_numpy(binarize_mask(mask_b)[None, None].astype(np.float32))

    dice = float(dice_metric(pred, target).item())
    hd95 = float(hd95_metric(pred, target).item())
    masd = float(masd_metric(pred, target).item())

    dice_metric.reset()
    hd95_metric.reset()
    masd_metric.reset()
    return dice, hd95, masd


def compute_monai_dice(mask_a, mask_b, dice_metric):
    pred = torch.from_numpy(binarize_mask(mask_a)[None, None].astype(np.float32))
    target = torch.from_numpy(binarize_mask(mask_b)[None, None].astype(np.float32))
    dice = float(dice_metric(pred, target).item())
    dice_metric.reset()
    return dice

def compute_monai_distances(mask_a, mask_b, hd95_metric, masd_metric):
    pred = torch.from_numpy(binarize_mask(mask_a)[None, None].astype(np.float32))
    target = torch.from_numpy(binarize_mask(mask_b)[None, None].astype(np.float32))
    hd95 = float(hd95_metric(pred, target).item())
    masd = float(masd_metric(pred, target).item())
    hd95_metric.reset()
    masd_metric.reset()
    return hd95, masd

def compute_masd_in_mm(masd_px, px_per_mm):
    if px_per_mm is None or px_per_mm == 0 or np.isnan(masd_px):
        return np.nan
    return masd_px / px_per_mm


def compute_hd95_in_mm(hd95_px, px_per_mm):
    if px_per_mm is None or px_per_mm == 0 or np.isnan(hd95_px):
        return np.nan
    return hd95_px / px_per_mm


def keep_largest_component(mask):
    mask = binarize_mask(mask).astype(bool)
    if not mask.any():
        return mask.astype(np.uint8)
    labeled, num = ndimage.label(mask)
    if num <= 1:
        return mask.astype(np.uint8)
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    largest = counts.argmax()
    return (labeled == largest).astype(np.uint8)


def load_pixels_per_mm_map(resolution_csv):
    df = pd.read_csv(resolution_csv)
    return {int(row["dataset_id"]): float(row["scale(pixels/mm)"]) for _, row in df.iterrows()}


def compute_geodesic_skeleton_length_mm(skeleton, scale_x, scale_y):
    coords = np.column_stack(np.nonzero(skeleton))
    if len(coords) == 0:
        return np.nan

    coord_to_idx = {tuple(coord): i for i, coord in enumerate(coords)}
    rows, cols, weights = [], [], []
    degrees = np.zeros(len(coords), dtype=int)

    for i, (y, x) in enumerate(coords):
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                neighbor = (y + dy, x + dx)
                j = coord_to_idx.get(neighbor)
                if j is None:
                    continue
                degrees[i] += 1
                if j > i:
                    dist = np.sqrt((dy * scale_y) ** 2 + (dx * scale_x) ** 2)
                    rows.extend([i, j])
                    cols.extend([j, i])
                    weights.extend([dist, dist])

    graph = csr_matrix((weights, (rows, cols)), shape=(len(coords), len(coords)))
    endpoints = np.where(degrees == 1)[0]

    if len(endpoints) >= 2:
        dists = shortest_path(graph, directed=False, indices=endpoints)
        endpoint_dists = dists[:, endpoints]
        finite = endpoint_dists[np.isfinite(endpoint_dists)]
    else:
        dists = shortest_path(graph, directed=False)
        finite = dists[np.isfinite(dists)]

    finite = finite[finite > 0]
    if finite.size == 0:
        return np.nan
    return float(finite.max())


def compute_cartilage_thickness(mask, px_per_mm):
    binary_mask = binarize_mask(mask)
    if not binary_mask.any() or px_per_mm is None or px_per_mm == 0:
        return np.nan

    scale_mm = 1.0 / px_per_mm
    skeleton = skeletonize(binary_mask.astype(bool))
    length_mm = compute_geodesic_skeleton_length_mm(skeleton, scale_mm, scale_mm)
    if length_mm == 0 or np.isnan(length_mm):
        return np.nan

    area_mm2 = binary_mask.sum() * (scale_mm ** 2)
    return area_mm2 / length_mm


def compute_echo_intensity(image, mask):
    binary_mask = binarize_mask(mask).astype(bool)
    if not binary_mask.any():
        return np.nan
    return float(np.mean(image[binary_mask]))


def format_metric_text(dice, masd_px, masd_mm, gt_thickness_mm=None, pred_thickness_mm=None):
    masd_px_text = "nan" if np.isnan(masd_px) else f"{masd_px:.2f}"
    masd_mm_text = "nan" if np.isnan(masd_mm) else f"{masd_mm:.2f}"
    lines = [
        f"Dice: {dice * 100:.2f}%",
        f"MASD: {masd_px_text} px",
        f"MASD: {masd_mm_text} mm",
    ]
    if gt_thickness_mm is not None:
        gt_text = "nan" if np.isnan(gt_thickness_mm) else f"{gt_thickness_mm:.2f}"
        lines.append(f"GT thickness: {gt_text} mm")
    if pred_thickness_mm is not None:
        pred_text = "nan" if np.isnan(pred_thickness_mm) else f"{pred_thickness_mm:.2f}"
        lines.append(f"Pred thickness: {pred_text} mm")
    return "\n".join(lines)


def plot_prediction_comparison(
    img, gt, pred, orig_image, orig_gt, orig_gt_from_pred_space, orig_pred, raw_metric_text, orig_metric_text
):
    plt.figure(figsize=(20, 8))

    ax = plt.subplot(1, 2, 1)
    ax.imshow(img, cmap="gray")
    ax.contour(gt, levels=[0.5], colors="yellow")
    ax.contour(pred, levels=[0.5], colors="red", linewidths=1.25)
    ax.text(
        0.02,
        0.02,
        raw_metric_text,
        transform=ax.transAxes,
        fontsize=10,
        family="monospace",
        color="white",
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="black", edgecolor="white", linewidth=1.0, alpha=0.8),
    )
    ax.axis("off")

    ax = plt.subplot(1, 2, 2)
    ax.imshow(orig_image, cmap="gray")
    ax.contour(orig_gt, levels=[0.5], colors="yellow")
    ax.contour(orig_gt_from_pred_space, levels=[0.5], colors="cyan", linewidths=1.0)
    ax.contour(orig_pred, levels=[0.5], colors="red", linewidths=1.25)
    ax.text(
        0.02,
        0.03,
        orig_metric_text,
        transform=ax.transAxes,
        fontsize=10,
        family="monospace",
        color="white",
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="black", edgecolor="white", linewidth=1.0, alpha=0.8),
    )
    ax.axis("off")

    plt.show()


def estimate_mask_transform(raw_mask, orig_mask):
    x_raw, y_raw, w_raw, h_raw = get_bounding_box(raw_mask)
    x_orig, y_orig, w_orig, h_orig = get_bounding_box(orig_mask)

    if w_raw == 0 or h_raw == 0:
        return {
            "scale_x": 1.0,
            "scale_y": 1.0,
            "tx": 0.0,
            "ty": 0.0,
            "raw_bbox": (x_raw, y_raw, w_raw, h_raw),
            "orig_bbox": (x_orig, y_orig, w_orig, h_orig),
        }

    scale_x = w_orig / w_raw
    scale_y = h_orig / h_raw
    tx = x_orig - scale_x * x_raw
    ty = y_orig - scale_y * y_raw
    return {
        "scale_x": scale_x,
        "scale_y": scale_y,
        "tx": tx,
        "ty": ty,
        "raw_bbox": (x_raw, y_raw, w_raw, h_raw),
        "orig_bbox": (x_orig, y_orig, w_orig, h_orig),
    }


def apply_mask_transform(mask, transform, out_w, out_h):
    if isinstance(mask, np.ndarray):
        mask = Image.fromarray(mask.astype(np.uint8))

    scaled_w = max(1, int(round(mask.size[0] * transform["scale_x"])))
    scaled_h = max(1, int(round(mask.size[1] * transform["scale_y"])))
    resized = mask.resize((scaled_w, scaled_h), resample=Image.NEAREST)
    resized_arr = np.array(resized)
    canvas = np.zeros((out_h, out_w), dtype=np.uint8)

    x0 = int(round(transform["tx"]))
    y0 = int(round(transform["ty"]))
    x1 = max(0, x0)
    y1 = max(0, y0)
    x2 = min(out_w, x0 + scaled_w)
    y2 = min(out_h, y0 + scaled_h)

    if x1 >= x2 or y1 >= y2:
        return canvas

    src_x1 = x1 - x0
    src_y1 = y1 - y0
    src_x2 = src_x1 + (x2 - x1)
    src_y2 = src_y1 + (y2 - y1)
    canvas[y1:y2, x1:x2] = resized_arr[src_y1:src_y2, src_x1:src_x2]
    return canvas


def resolve_original_paths(orig_datasets, test_dataset_id, pred_path):
    fn = os.path.basename(pred_path)
    if test_dataset_id == 79:
        orig_folder = "KneeUS_Ilker"
        orig_fn = fn
    else:
        orig_folder, orig_fn = fn.split("___")
    orig_fn = orig_fn.rsplit("_", 1)[0]
    orig_image_path = glob(f"{orig_datasets}/{orig_folder}/images/{orig_fn}*")[0]
    orig_gt_path = glob(f"{orig_datasets}/{orig_folder}/masks/{orig_fn}*")[0]
    return orig_image_path, orig_gt_path


def load_completed_ids(save_path):
    if not os.path.exists(save_path):
        return set()
    df = pd.read_csv(save_path, usecols=["image_id"])
    return set(df["image_id"].dropna().astype(str).tolist())


def main():
    args = parse_args()

    nnunet_raw = os.environ["nnUNet_raw"]
    os.environ["nnUNet_results"] = os.environ.get("nnUNet_results", "")

    train_dataset = get_dataset_name(nnunet_raw, args.train_dataset_id)
    test_dataset = get_dataset_name(nnunet_raw, args.test_dataset_id)
    split = args.split or ("Ts" if args.test_dataset_id == 79 else "Tr")

    model_dir = os.path.join(args.models_dir, args.model, train_dataset, f"fold_{args.fold}")
    preds_dir = f"{model_dir}/test/{test_dataset}/preds"
    if test_dataset == train_dataset:
        image_wise_results_path = os.path.join(
            os.path.dirname(model_dir), f"image_wise_results_largest_component_{test_dataset}.csv"
        )
        save_path = f"{os.path.dirname(model_dir)}/results_{test_dataset}.csv"
        outcomes_save_path = f"{os.path.dirname(model_dir)}/cartilage_outcomes_{test_dataset}.csv"
    else:
        save_path = f"{model_dir}/test/results_{test_dataset}.csv"
        outcomes_save_path = f"{model_dir}/test/cartilage_outcomes_{test_dataset}.csv"
        image_wise_results_path = os.path.join(
            model_dir, "test", f"image_wise_results_largest_component_{test_dataset}.csv"
        )

    images_dir = f"{nnunet_raw}/{test_dataset}/images{split}"
    gt_dir = f"{nnunet_raw}/{test_dataset}/labels{split}"

    px_per_mm_map = load_pixels_per_mm_map(args.resolution_csv)
    compute_performance = args.compute == "performance"
    compute_outcomes = args.compute == "outcomes"

    if compute_performance:
        image_wise_results = pd.read_csv(image_wise_results_path)
        image_wise_results = image_wise_results.drop_duplicates(subset="image_id").set_index("image_id")
        dice_metric = DiceMetric(include_background=False, reduction="mean")
        hd95_metric = HausdorffDistanceMetric(include_background=False, reduction="mean", percentile=95)
        masd_metric = SurfaceDistanceMetric(include_background=False, reduction="mean")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if args.recompute_existing:
        completed_ids = set()
        file_mode = "w"
    else:
        active_save_path = save_path if compute_performance else outcomes_save_path
        completed_ids = load_completed_ids(active_save_path)
        file_mode = "a" if os.path.exists(active_save_path) else "w"

    active_save_path = save_path if compute_performance else outcomes_save_path
    with open(active_save_path, file_mode, newline="") as csvfile:
        if compute_performance:
            writer = csv.DictWriter(csvfile, fieldnames=["image_id", "dice", "hd95", "masd"])
        else:
            writer = csv.DictWriter(
                csvfile,
                fieldnames=["image_id", "gt_thickness", "pred_thickness", "gt_echo_intensity", "pred_echo_intensity"],
            )
        if file_mode == "w":
            writer.writeheader()

        files = glob(join(preds_dir, "*.png"))
        print(f"Found {len(files)} files in {preds_dir}")
        if completed_ids:
            print(f"Skipping {len(completed_ids)} completed image_ids from {active_save_path}")
        if args.debug or args.shuffle:
            random.shuffle(files)

        for pred_path in tqdm(files):
            fn = os.path.basename(pred_path)
            stem = os.path.splitext(fn)[0]
            if fn in IGNORE_IDS or stem in IGNORE_IDS:
                continue
            if stem in completed_ids:
                continue
            if compute_performance and stem not in image_wise_results.index:
                continue

            image_path = f"{images_dir}/{fn}".replace(".png", "_0000.png")
            gt_path = f"{gt_dir}/{fn}"
            orig_image_path, orig_gt_path = resolve_original_paths(args.orig_datasets, args.test_dataset_id, pred_path)

            gt = binarize_mask(np.array(Image.open(gt_path)))
            pred = binarize_mask(np.array(Image.open(pred_path)))
            gt_in_pred_space = binarize_mask(np.array(Image.fromarray(gt.astype(np.uint8)).resize((pred.shape[1], pred.shape[0]), resample=Image.NEAREST)))
            orig_image = np.array(Image.open(orig_image_path).convert("L"))
            orig_gt = binarize_mask(np.array(Image.open(orig_gt_path)))

            transform = estimate_mask_transform(gt_in_pred_space, orig_gt)
            orig_pred = apply_mask_transform(pred, transform, orig_gt.shape[1], orig_gt.shape[0])

            orig_px_per_mm = px_per_mm_map[args.test_dataset_id]

            if compute_performance:
                source_row = image_wise_results.loc[stem]
                dice = float(source_row["dice"])
                hd95_px, masd_px = compute_monai_distances(orig_pred, orig_gt, hd95_metric, masd_metric)
                hd95_mm = compute_hd95_in_mm(hd95_px, orig_px_per_mm)
                masd_mm = compute_masd_in_mm(masd_px, orig_px_per_mm)
            else:
                gt_thickness_mm = compute_cartilage_thickness(orig_gt, orig_px_per_mm)
                pred_thickness_mm = compute_cartilage_thickness(orig_pred, orig_px_per_mm)
                gt_echo_intensity = compute_echo_intensity(orig_image, orig_gt)
                pred_echo_intensity = compute_echo_intensity(orig_image, orig_pred)

            if args.debug:
                img = np.array(Image.open(image_path).resize((pred.shape[1], pred.shape[0]), resample=Image.NEAREST))
                orig_gt_from_pred_space = apply_mask_transform(
                    gt_in_pred_space, transform, orig_gt.shape[1], orig_gt.shape[0]
                )
                pred_px_per_mm = orig_px_per_mm / np.sqrt(transform["scale_x"] * transform["scale_y"])
                if compute_performance:
                    dice_pred, _, masd_pred_px = compute_monai_metrics(
                        pred, gt_in_pred_space, dice_metric, hd95_metric, masd_metric
                    )
                    masd_pred_mm = compute_masd_in_mm(masd_pred_px, pred_px_per_mm)
                    dice_orig, _, masd_orig_px = compute_monai_metrics(
                        orig_pred, orig_gt, dice_metric, hd95_metric, masd_metric
                    )
                    masd_orig_mm = compute_masd_in_mm(masd_orig_px, orig_px_per_mm)
                    raw_metric_text = format_metric_text(dice_pred, masd_pred_px, masd_pred_mm)
                    orig_metric_text = format_metric_text(
                        dice_orig, masd_orig_px, masd_orig_mm, gt_thickness_mm, pred_thickness_mm
                    )
                else:
                    gt_thickness_mm = compute_cartilage_thickness(orig_gt, orig_px_per_mm)
                    pred_thickness_mm = compute_cartilage_thickness(orig_pred, orig_px_per_mm)
                    raw_metric_text = format_metric_text(np.nan, np.nan, np.nan)
                    orig_metric_text = format_metric_text(
                        np.nan, np.nan, np.nan, gt_thickness_mm=gt_thickness_mm, pred_thickness_mm=pred_thickness_mm
                    )
                plot_prediction_comparison(
                    img,
                    gt_in_pred_space,
                    pred,
                    orig_image,
                    orig_gt,
                    orig_gt_from_pred_space,
                    orig_pred,
                    raw_metric_text,
                    orig_metric_text,
                )
                return

            if compute_performance:
                writer.writerow(
                    {
                        "image_id": stem,
                        "dice": round(dice, 2),
                        "hd95": round(hd95_mm, 4) if not np.isnan(hd95_mm) else np.nan,
                        "masd": round(masd_mm, 4) if not np.isnan(masd_mm) else np.nan,
                    }
                )
            else:
                writer.writerow(
                    {
                        "image_id": stem,
                        "gt_thickness": round(gt_thickness_mm, 4) if not np.isnan(gt_thickness_mm) else np.nan,
                        "pred_thickness": round(pred_thickness_mm, 4) if not np.isnan(pred_thickness_mm) else np.nan,
                        "gt_echo_intensity": round(gt_echo_intensity, 4) if not np.isnan(gt_echo_intensity) else np.nan,
                        "pred_echo_intensity": round(pred_echo_intensity, 4) if not np.isnan(pred_echo_intensity) else np.nan,
                    }
                )
            csvfile.flush()
            completed_ids.add(stem)

    if compute_performance:
        print(f"Saved results to {save_path}")
    if compute_outcomes:
        print(f"Saved cartilage outcomes to {outcomes_save_path}")


if __name__ == "__main__":
    main()
