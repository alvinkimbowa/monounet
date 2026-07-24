import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm


TOOLS_ROOT = Path("/home/ultrai/UltrAi/knee_us_segmentation/tools_image-review")
IMAGEJ_DATA_ROOT = TOOLS_ROOT / "ImageJ - Knee US Cartilage Segmentation" / "data"
MASTER_LIST_PATH = TOOLS_ROOT / "master_list.json"
ANNOTATIONS_ROOT = IMAGEJ_DATA_ROOT / "annotations"
IMAGES_ROOT = IMAGEJ_DATA_ROOT / "images"
RAW_DATASETS_ROOT = Path("/home/ultrai/UltrAi/knee_us_segmentation/data/raw_data/datasets")
REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "interrater"
INCLUDED_SESSIONS = ("Session 1", "Session 2", "Session 3", "Session 4")
ANNOTATORS = ("Will", "Ibrahim")


def load_master_list(master_list_path):
    with open(master_list_path, "r") as f:
        master_list = json.load(f)
    return {str(k): Path(v).stem for k, v in master_list.items()}


def build_duplicate_lookup(master_list):
    counts = Counter(master_list.values())
    occurrence_counts = defaultdict(int)
    duplicate_lookup = {}

    for anonymized_id, original_image_id in sorted(master_list.items(), key=lambda x: int(x[0])):
        duplicate_lookup[anonymized_id] = {
            "is_duplicate_image": counts[original_image_id] > 1,
            "duplicate_index": occurrence_counts[original_image_id],
        }
        occurrence_counts[original_image_id] += 1

    return duplicate_lookup


def read_imagej_polygon_roi(roi_path):
    data = roi_path.read_bytes()
    if data[:4] != b"Iout":
        raise ValueError(f"Invalid ImageJ ROI magic in {roi_path}")

    roi_type = data[6]
    if roi_type != 0:
        raise ValueError(f"Unsupported ROI type {roi_type} in {roi_path}; expected polygon type 0")

    top = int.from_bytes(data[8:10], "big", signed=True)
    left = int.from_bytes(data[10:12], "big", signed=True)
    n_coords = int.from_bytes(data[16:18], "big")

    coord_offset = 64
    x_bytes = data[coord_offset:coord_offset + 2 * n_coords]
    y_bytes = data[coord_offset + 2 * n_coords:coord_offset + 4 * n_coords]
    x_rel = np.frombuffer(x_bytes, dtype=">u2").astype(np.int32)
    y_rel = np.frombuffer(y_bytes, dtype=">u2").astype(np.int32)
    coords = np.stack([x_rel + left, y_rel + top], axis=1)
    return coords


def rasterize_roi(roi_path, image_shape):
    coords = read_imagej_polygon_roi(roi_path)
    mask_image = Image.new("L", (image_shape[1], image_shape[0]), 0)
    draw = ImageDraw.Draw(mask_image)
    draw.polygon([tuple(point) for point in coords.astype(np.int32)], outline=1, fill=1)
    return np.array(mask_image, dtype=np.uint8)


def to_label_style_image_id(original_image_id):
    prefix_map = [
        "ACLMRI_Michel_",
        "Tufts_6m_Jamil_",
        "Tufts_3m_Sunny_",
        "Tufts_preop_Harkey_",
        "CaseControl_ACL_",
        "CaseControl_Control_",
        "Nike_Arjun_clarius_auto_",
        "Nike_Arjun_clarius_std_",
        "Nike_Arjun_ge_auto_",
        "Nike_Arjun_ge_std_",
        "Nike_Brian_",
        "Nike_Michel_",
        "Nike_Olivia_",
        "Nike_Jagger_",
        "Nike_XC_Jamil_",
    ]
    for prefix in prefix_map:
        if original_image_id.startswith(prefix):
            return original_image_id.replace(prefix, f"{prefix[:-1]}___", 1)
    return original_image_id


def get_export_name(original_image_id, duplicate_index):
    if duplicate_index > 0:
        return None
    return f"{to_label_style_image_id(original_image_id)}.png"


def normalize_review_image_id(stem):
    prefixes = [
        "ACLMRI_Michel_",
        "Tufts_6m_Jamil_",
        "Tufts_3m_Sunny_",
        "Tufts_preop_Harkey_",
        "CaseControl_ACL_",
        "CaseControl_Control_",
        "Nike_Arjun_clarius_auto_",
        "Nike_Arjun_clarius_std_",
        "Nike_Arjun_ge_auto_",
        "Nike_Arjun_ge_std_",
        "Nike_Brian_",
        "Nike_Michel_",
        "Nike_Olivia_",
        "Nike_Jagger_",
        "Nike_XC_Jamil_",
    ]
    for prefix in prefixes:
        if stem.startswith(prefix):
            return stem[len(prefix):].lower()
    return stem.lower()


def normalize_mask_name(path):
    stem = Path(path).name
    if stem.endswith(".png"):
        stem = stem[:-4]
    if stem.endswith(".tif"):
        stem = stem[:-4]
    if stem.endswith("_mask"):
        stem = stem[:-5]
    return stem.lower()


def build_expert_mask_lookup(raw_datasets_root):
    mask_map = defaultdict(list)
    for mask_path in raw_datasets_root.glob("*/masks/*"):
        mask_map[normalize_mask_name(mask_path)].append(mask_path)
    return mask_map


def choose_expert_mask_path(original_image_id, candidate_paths):
    if len(candidate_paths) == 1:
        return candidate_paths[0]

    original_lower = original_image_id.lower()
    preferred_tokens = []
    if "nike_brian_" in original_lower:
        preferred_tokens.append("/Nike_Brian/")
    if "nike_michel_" in original_lower:
        preferred_tokens.append("/Nike_Michel/")
    if "nike_olivia_" in original_lower:
        preferred_tokens.append("/Nike_Olivia/")
    if "nike_jagger_" in original_lower:
        preferred_tokens.append("/Nike_Jagger/")
    if "nike_xc_jamil_" in original_lower:
        preferred_tokens.append("/Nike_XC_Jamil/")
    if "nike_arjun_clarius_" in original_lower:
        preferred_tokens.append("/Nike_Arjun_clarius_")
    if "nike_arjun_ge_" in original_lower:
        preferred_tokens.append("/Nike_Arjun_ge_")
    if "aclmri_michel_" in original_lower:
        preferred_tokens.append("/ACLMRI_Michel/")
    if "casecontrol_acl_" in original_lower:
        preferred_tokens.append("/CaseControl_ACL/")
    if "casecontrol_control_" in original_lower:
        preferred_tokens.append("/CaseControl_Control/")
    if "tufts_3m_sunny_" in original_lower:
        preferred_tokens.append("/Tufts_3m_Sunny/")
    if "tufts_6m_jamil_" in original_lower:
        preferred_tokens.append("/Tufts_6m_Jamil/")
    if "tufts_preop_harkey_" in original_lower:
        preferred_tokens.append("/Tufts_preop_Harkey/")

    for token in preferred_tokens:
        for candidate_path in candidate_paths:
            if token in str(candidate_path):
                return candidate_path

    raise ValueError(f"Ambiguous expert mask matches for {original_image_id}: {candidate_paths}")


def get_expert_mask_bounds(original_image_id, expert_mask_lookup):
    normalized_id = normalize_review_image_id(original_image_id)
    candidate_paths = expert_mask_lookup.get(normalized_id, [])
    if not candidate_paths:
        raise FileNotFoundError(f"No expert mask found for {original_image_id}")

    expert_mask_path = choose_expert_mask_path(original_image_id, candidate_paths)
    expert_mask = np.array(Image.open(expert_mask_path).convert("L")) > 0
    cols = np.where(expert_mask.sum(axis=0) > 0)[0]
    if len(cols) == 0:
        raise ValueError(f"Expert mask has no foreground: {expert_mask_path}")
    return int(cols[0]), int(cols[-1]), expert_mask_path


def export_interrater_masks(
    output_root,
    master_list_path=MASTER_LIST_PATH,
    annotations_root=ANNOTATIONS_ROOT,
    images_root=IMAGES_ROOT,
    raw_datasets_root=RAW_DATASETS_ROOT,
):
    master_list = load_master_list(master_list_path)
    duplicate_lookup = build_duplicate_lookup(master_list)
    expert_mask_lookup = build_expert_mask_lookup(raw_datasets_root)

    output_root.mkdir(parents=True, exist_ok=True)
    for annotator in ANNOTATORS:
        (output_root / annotator).mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    missing_paths = []

    for session in INCLUDED_SESSIONS:
        session_image_dir = images_root / session
        if not session_image_dir.is_dir():
            raise FileNotFoundError(f"Missing session image directory: {session_image_dir}")

        image_paths = sorted(session_image_dir.glob("*.tif"), key=lambda p: int(p.stem))
        for image_path in tqdm(image_paths, desc=f"Exporting {session}", leave=False):
            anonymized_id = image_path.stem
            if anonymized_id not in master_list:
                raise KeyError(f"Missing anonymized id {anonymized_id} in {master_list_path}")

            original_image_id = master_list[anonymized_id]
            image_shape = np.array(Image.open(image_path).convert("L")).shape[:2]
            dup_info = duplicate_lookup[anonymized_id]
            export_name = get_export_name(original_image_id, dup_info["duplicate_index"])
            if export_name is None:
                continue

            left_end, right_end, expert_mask_path = get_expert_mask_bounds(original_image_id, expert_mask_lookup)

            for annotator in ANNOTATORS:
                roi_path = annotations_root / annotator / session / f"{anonymized_id}.roi"
                if not roi_path.exists():
                    missing_paths.append(str(roi_path))
                    continue

                mask = rasterize_roi(roi_path, image_shape)
                mask[:, :left_end] = 0
                mask[:, right_end + 1:] = 0

                out_path = output_root / annotator / export_name
                Image.fromarray(mask).save(out_path)

                manifest_rows.append(
                    {
                        "annotator": annotator,
                        "session": session,
                        "anonymized_id": anonymized_id,
                        "original_image_id": original_image_id,
                        "label_style_image_id": to_label_style_image_id(original_image_id),
                        "is_duplicate_image": dup_info["is_duplicate_image"],
                        "duplicate_index": dup_info["duplicate_index"],
                        "export_name": export_name,
                        "export_path": str(out_path),
                        "expert_mask_path": str(expert_mask_path),
                        "crop_left": left_end,
                        "crop_right": right_end,
                    }
                )

    if missing_paths:
        missing_paths = sorted(set(missing_paths))
        raise FileNotFoundError("Missing ROI files:\n" + "\n".join(missing_paths))

    manifest_path = output_root / "interrater_manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "annotator",
                "session",
                "anonymized_id",
                "original_image_id",
                "label_style_image_id",
                "is_duplicate_image",
                "duplicate_index",
                "export_name",
                "export_path",
                "expert_mask_path",
                "crop_left",
                "crop_right",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    return manifest_path, len(manifest_rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess interrater ROI annotations into masks.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where the exported masks and manifest will be saved.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    manifest_path, num_masks = export_interrater_masks(args.output_root)
    print(f"Saved {num_masks} masks.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
