import argparse
import os
from glob import glob

import cv2
import csv
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import label

import archs
from dataset import nnUNetDataset
from metrics import iou_score
from utils import AverageMeter, str2bool
from albumentations import RandomRotate90,Resize
import time
from archs import UNext
import numpy as np
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric
from TinyUNet import TinyUNet
from monounet import MonogenicNets, MonoUNets
from monounet.MonogenicNets import center_crop

MONO_ARCH_NAMES = MonogenicNets.__all__
MONOUNET_ARCH_NAMES = MonoUNets.__all__

def visualize_prediction(img, pred, gt=None, dice=None, masd=None, hd95=None, save_path=None):
    """
    Visualize prediction overlaid on input image with optional ground truth and metrics.
    
    Args:
        img: Input image as numpy array (H, W) for grayscale or (H, W, 3) for RGB
        pred: Prediction mask as numpy array (H, W) with values 0 or 1
        gt: Optional ground truth mask as numpy array (H, W) with values 0 or 1
        dice: Optional dice score (0-100)
        masd: Optional MASD score
        hd95: Optional HD95 score
        save_path: Path to save the visualized image
    """
    plt.figure(figsize=(4, 4))
    plt.imshow(img, cmap='gray')
    
    if gt is not None and gt.max() > 0:
        plt.contour(gt, colors='#90EE90', linewidths=1)
    
    # Overlay prediction (red)
    plt.contour(pred, colors='#FF0000', linewidths=1)
    
    # Add metrics text at bottom left
    metrics_text = []
    if dice is not None:
        metrics_text.append(f'Dice: {dice*100:.2f}%')
    if masd is not None:
        metrics_text.append(f'MASD: {masd:.2f}')
    if hd95 is not None:
        metrics_text.append(f'HD95: {hd95:.2f}')
        
    text_str = '\n'.join(metrics_text)
    img_height, img_width = img.shape[:2]
        
    plt.text(img_width * 0.02, img_height * 0.98, text_str,
            fontsize=10, color='white',
            verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='black'))
    
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        print("Saving prediction to: ", save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', required=True,
                        help='model name')
    parser.add_argument('--cascade', default=False, type=str2bool,
                        help='use cascade (True or False)')
    parser.add_argument('--refiner_arch', default='MonoUNetBase', type=str,
                        help='refiner architecture')
    parser.add_argument('--train_dataset', default='Dataset073_GE_LE',
                        help='train dataset name')
    parser.add_argument('--train_fold', type=str, required=True,
                        help='train fold index (0-4) or "all" to combine all folds')
    parser.add_argument('--test_dataset', type=str, required=True,
                        help='test dataset name')
    parser.add_argument('--test_split', type=str, required=True,
                        help='test split (Ts or Te)')
    parser.add_argument('--save_preds', type=str2bool, default=False,
                        help='save predictions (True or False)')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='model_best.pth')
    parser.add_argument('--deep_supervision', default=False, type=str2bool,
                        help='use deep supervision (affects model directory path)')
    parser.add_argument('--data_augmentation', default=True, type=str2bool,
                        help='use data augmentation (affects model directory path)')
    parser.add_argument('--overlay', type=str2bool, default=False,
                        help='overlay predictions on images and save visualized images (True or False)')
    parser.add_argument('--largest_component', type=str2bool, default=False,
                        help='keep only the largest connected component (True or False)')
    args = parser.parse_args()

    return args

def find_largest_component_per_class(segmentation, num_classes):
    output = np.zeros_like(segmentation)
    for i in range(segmentation.shape[0]):
        for cls in range(1, num_classes):  # skip background class 0
            binary_mask = (segmentation[i] == cls).astype(np.uint8)
            labeled_array, num_features = label(binary_mask)
            if num_features == 0:
                continue
            largest_label = max(range(1, num_features + 1), key=lambda x: np.sum(labeled_array == x))
            output[i][labeled_array == largest_label] = cls
    return output

def main():
    args = parse_args()

    arch_name = args.name
    if args.deep_supervision:
        arch_name += 'DS'
    if args.data_augmentation:
        arch_name += 'DA'
    base_arch_name = arch_name
    if args.cascade:
        arch_name += 'Cascade'
    model_dir = f"models/{arch_name}/{args.train_dataset}/fold_{args.train_fold}"
    base_model_dir = f"models/{base_arch_name}/{args.train_dataset}/fold_{args.train_fold}"
    with open(f'{model_dir}/config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    print("=> creating model %s" % config['arch'])
    if config['arch'] == "UNext" or config['arch'] == "UNext_S":
        model = archs.__dict__[config['arch']](config['num_classes'],
                                               config['input_channels'],
                                               deep_supervision=False)
    elif config['arch'] == "TinyUNet":
        model = TinyUNet(config['input_channels'],
                         config['num_classes'])
    elif config['arch'] in MONO_ARCH_NAMES:
        model = MonogenicNets.__dict__[config['arch']](config['input_channels'],
                                                                config['num_classes'],
                                                                img_size=(config['input_h'], config['input_w']),
                                                                deep_supervision=False)
    elif config['arch'] in MONOUNET_ARCH_NAMES:
        model = MonoUNets.__dict__[config['arch']](in_channels=config['input_channels'],
                                                                num_classes=config['num_classes'],
                                                                img_size=(config['input_h'], config['input_w']),
                                                                deep_supervision=False)
    else:
        raise NotImplementedError
    
    if args.cascade:
        model = MonoUNets.CascadeBase(
            base_arch=MonoUNets.__dict__[config['arch']],
            refiner_arch=MonoUNets.__dict__[args.refiner_arch],
            base_ckpt=os.path.join(base_model_dir, args.ckpt),
            base_kwargs={"in_channels": config['input_channels'], "num_classes": config['num_classes'], "img_size": (config['input_h'], config['input_w']), "deep_supervision": False},
            refiner_kwargs={"in_channels": config['input_channels'] + 1, "num_classes": config['num_classes'], "img_size": (config['input_h'], config['input_w']), "deep_supervision": False},
        )
    
    model = model.cuda()

    ckpt_path = os.path.join(model_dir, args.ckpt)
    ckpt_data = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = ckpt_data.get('model_state_dict', ckpt_data.get('network_weights', ckpt_data))
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    if args.test_split == 'Ts':
        val_dataset = nnUNetDataset(
            dataset_name=args.test_dataset,
            input_channels=config['input_channels'],
            num_classes=config['num_classes'],
            split=args.test_split,
            transform=val_transform,
            eval=True)
    elif args.test_dataset == args.train_dataset:
        assert args.train_fold != 'all', "TODO: Implement validation for all folds"
        val_dataset = nnUNetDataset(
            dataset_name=args.test_dataset,
            input_channels=config['input_channels'],
            num_classes=config['num_classes'],
            split=args.test_split,
            fold=args.train_fold,
            split_type='val',
            transform=val_transform,
            eval=False)
    else:
        val_dataset = nnUNetDataset(
            dataset_name=args.test_dataset,
            input_channels=config['input_channels'],
            num_classes=config['num_classes'],
            split=args.test_split,
            transform=val_transform,
            eval=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    iou_avg_meter = AverageMeter()
    dice_avg_meter = AverageMeter()
    gput = AverageMeter()
    cput = AverageMeter()

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    hd95_metric = HausdorffDistanceMetric(include_background=False, reduction="mean", percentile=95)
    surface_dice_metric = SurfaceDistanceMetric(include_background=False, reduction="mean")

    if args.save_preds:
        save_dir = os.path.join(model_dir, 'test', args.test_dataset, 'preds')
        os.makedirs(save_dir, exist_ok=True)
    
    if args.overlay and args.save_preds:
        overlay_dir = os.path.join(model_dir, 'test', args.test_dataset, 'overlays')
        os.makedirs(overlay_dir, exist_ok=True)
    
    # Collect image IDs to match with metrics later
    image_ids = []        
    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            model = model.cuda()

            # check if target is 0,1.. if it is 0,255 then convert it to 0,1
            if target.max() == 255:
                target = (target / 255).long()

            # compute output
            output = model(input).cpu()

            output = torch.sigmoid(output)
            output[output>=0.5]=1
            output[output<0.5]=0

            if args.largest_component:
                output = output.cpu().numpy()
                output = find_largest_component_per_class(output, config['num_classes'] + 1)
                output = torch.from_numpy(output)

                target = target.cpu().numpy()
                target = find_largest_component_per_class(target, config['num_classes'] + 1)
                target = torch.from_numpy(target)

            target = target.unsqueeze(1)

            dice = dice_metric(output, center_crop(target, output.shape[2:]))
            hd95 = hd95_metric(output, center_crop(target, output.shape[2:]))
            masd = surface_dice_metric(output, center_crop(target, output.shape[2:]))

            output = output.numpy()
            
            # Collect image IDs
            for i in range(len(output)):
                image_ids.append(meta['img_id'][i])

            if args.save_preds:
                for i in range(len(output)):
                    for c in range(config['num_classes']):
                        if args.overlay:
                            save_path = os.path.join(overlay_dir, meta['img_id'][i] + '.png')
                            visualize_prediction(
                                img=input[i, 0, :, :].cpu().numpy(),
                                gt=target[i, c].cpu().numpy(),
                                pred=output[i, c],
                                dice=dice[i].item(),
                                masd=masd[i].item(),
                                hd95=hd95[i].item(),
                                save_path=save_path
                            )
                        else:
                            cv2.imwrite(os.path.join(save_dir, meta['img_id'][i] + '.png'),
                                    (output[i, c] * 255).astype('uint8'))
            

    torch.cuda.empty_cache()

    if args.save_preds and args.overlay:
        print("Overlay mode is enabled. Metrics will not be calculated.")
        return
    
    # Calculate metrics
    dice_score = dice_metric.aggregate().item() * 100
    dice_std = dice_metric.get_buffer().std().item() * 100
    hd95_score = hd95_metric.aggregate().item()
    hd95_std = hd95_metric.get_buffer().std().item()
    masd_score = surface_dice_metric.aggregate().item()
    masd_std = surface_dice_metric.get_buffer().std().item()

    # Print metrics
    print("\n")
    print(f"Dice: {dice_score:.2f}% ± {dice_std:.2f}%")
    print(f"HD95: {hd95_score:.2f} ± {hd95_std:.2f}")
    print(f"MASD: {masd_score:.2f} ± {masd_std:.2f}")

    # Save aggregated metrics to CSV
    results_csv_path = os.path.join(model_dir, 'test', f'results{"_largest_component" if args.largest_component else ""}.csv')
    os.makedirs(os.path.dirname(results_csv_path), exist_ok=True)
    csv_exists = os.path.exists(results_csv_path) and os.path.getsize(results_csv_path) > 0
    with open(results_csv_path, 'a', newline='') as csvfile:
        fieldnames = ['test_dataset_name', 'dice', 'dice_std', 'hd95', 'hd95_std', 'masd', 'masd_std']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header if file doesn't exist
        if not csv_exists:
            writer.writeheader()
        
        # Write results
        writer.writerow({
            'test_dataset_name': args.test_dataset,
            'dice': f"{dice_score:.2f}",
            'dice_std': f"{dice_std:.2f}",
            'hd95': f"{hd95_score:.2f}",
            'hd95_std': f"{hd95_std:.2f}",
            'masd': f"{masd_score:.2f}",
            'masd_std': f"{masd_std:.2f}"
        })
    
    print(f"Results saved to: {results_csv_path}\n")
    
    # Save image-wise metrics to CSV (extract from metric buffers)
    if args.test_dataset == args.train_dataset:
        image_wise_csv_path = os.path.join(os.path.dirname(model_dir), f'image_wise_results{"_largest_component" if args.largest_component else ""}_{args.test_dataset}.csv')
    else:
        image_wise_csv_path = os.path.join(model_dir, 'test', f'image_wise_results{"_largest_component" if args.largest_component else ""}_{args.test_dataset}.csv')
    
    os.makedirs(os.path.dirname(image_wise_csv_path), exist_ok=True)
    
    # Extract per-image metrics from metric buffers
    dice_buffer = dice_metric.get_buffer() * 100  # Convert to percentage
    hd95_buffer = hd95_metric.get_buffer()
    masd_buffer = surface_dice_metric.get_buffer()
    
    csv_exists = os.path.exists(image_wise_csv_path) and os.path.getsize(image_wise_csv_path) > 0
    with open(image_wise_csv_path, 'a', newline='') as csvfile:
        fieldnames = ['image_id', 'dice', 'hd95', 'masd']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header only if file doesn't exist or is empty
        if not csv_exists:
            writer.writeheader()
        
        # Write image-wise results
        for i, image_id in enumerate(image_ids):
            writer.writerow({
                'image_id': image_id,
                'dice': f"{dice_buffer[i].item():.2f}",
                'hd95': f"{hd95_buffer[i].item():.2f}",
                'masd': f"{masd_buffer[i].item():.2f}"
            })
    
    print(f"Image-wise results saved to: {image_wise_csv_path}\n")


if __name__ == '__main__':
    main()
