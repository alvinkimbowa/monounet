import os
import json
import cv2
import numpy as np
import torch
import torch.utils.data
from glob import glob
import nibabel as nib


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):
        """
        Args:
            img_ids (list): Image ids.
            img_dir: Image file directory.
            mask_dir: Mask file directory.
            img_ext (str): Image file extension.
            mask_ext (str): Mask file extension.
            num_classes (int): Number of classes.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.
        
        Note:
            Make sure to put the files as the following structure:
            <dataset name>
            ├── images
            |   ├── 0a7e06.jpg
            │   ├── 0aab0a.jpg
            │   ├── 0b1761.jpg
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                |
                ├── 1
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                ...
        """
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        
        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))

        mask = []
        for i in range(self.num_classes):
            mask.append(cv2.imread(os.path.join(self.mask_dir, str(i),
                        img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])
        mask = np.dstack(mask)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        
        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)
        mask = mask.astype('float32') / 255
        mask = mask.transpose(2, 0, 1)
        
        return img, mask, {'img_id': img_id}


class nnUNetDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name, input_channels, num_classes, split, fold=None, split_type='train', transform=None, eval=False, oversample_foreground_percent=0.33):
        self.transform = transform
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.eval = eval
        self.split_type = split_type
        self.oversample_foreground_percent = oversample_foreground_percent
        
        nnunet_raw = os.environ['nnUNet_raw']
        nnunet_preprocessed = os.environ['nnUNet_preprocessed']
        self.img_dir = os.path.join(f'{nnunet_raw}/{dataset_name}/images{split}')
        self.label_dir = os.path.join(f'{nnunet_raw}/{dataset_name}/labels{split}')
        
        with open(os.path.join(f'{nnunet_raw}/{dataset_name}/dataset.json'), 'r') as f:
            dataset_info = json.load(f)
        self.img_ext = dataset_info['file_ending']
        
        if self.eval:
            img_ids = glob(f'{self.img_dir}/*{self.img_ext}')
            parsed_ids = []
            for img_id in img_ids:
                base = os.path.basename(img_id)
                stem = base[:-len(self.img_ext)]
                parts = stem.rsplit('_', 1)
                if len(parts) == 2 and parts[1].isdigit():
                    parsed_ids.append(parts[0])
                else:
                    parsed_ids.append(stem)
            self.img_ids = sorted(list(set(parsed_ids)))
        else:
            with open(os.path.join(f'{nnunet_preprocessed}/{dataset_name}/splits_final.json'), 'r') as f:
                splits = json.load(f)
            
            if fold == 'all':
                img_ids = []
                for split_dict in splits:
                    img_ids.extend(split_dict[split_type])
                img_ids = list(set(img_ids))
            else:
                img_ids = splits[int(fold)][split_type]
            
            self.img_ids = img_ids
        
        print("Found %d images" % len(self.img_ids))
    
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_filename = f'{self.img_dir}/{img_id}_0000{self.img_ext}'
        label_filename = f'{self.label_dir}/{img_id}{self.img_ext}'
        
        # Determine if other channels exist
        other_chs = [ch for ch in glob(f'{self.img_dir}/{img_id}_*{self.img_ext}') if ch != img_filename]
        other_chs = sorted(other_chs)

        if self.img_ext in ['.nii.gz', '.nii']:
            imgs = [nib.load(img_filename).get_fdata()]
            for ch in other_chs:
                imgs.append(nib.load(ch).get_fdata())

            seg_vol = None
            if os.path.exists(label_filename):
                seg_vol = nib.load(label_filename).get_fdata()
            elif not self.eval:
                raise ValueError(f"Label file not found: {label_filename}")

            force_fg = None
            if self.eval or self.split_type != 'train':
                force_fg = False

            _, seg_slice, z_idx, _ = sample_2d_slice_from_hwd_nnunet_style(
                imgs[0],
                seg_vol,
                oversample_foreground_percent=self.oversample_foreground_percent,
                force_fg=force_fg
            )

            stacked = []
            for vol in imgs:
                if vol.ndim == 3:
                    sl = vol[:, :, z_idx]
                else:
                    sl = vol
                stacked.append(sl[..., None].astype('float32'))
            img = np.concatenate(stacked, axis=-1)
            if self.input_channels == 3 and img.shape[-1] == 1:
                img = np.repeat(img, 3, axis=-1)
        else:
            # Dynamically load all available channels
            if self.input_channels == 3 and len(other_chs) == 0:
                img = cv2.imread(img_filename)
            else:
                imgs = [cv2.imread(img_filename, cv2.IMREAD_GRAYSCALE)[..., None]]
                for ch in other_chs:
                    imgs.append(cv2.imread(ch, cv2.IMREAD_GRAYSCALE)[..., None])
                img = np.concatenate(imgs, axis=-1)
        
        if os.path.exists(label_filename):
            if self.img_ext in ['.nii.gz', '.nii']:
                mask = seg_slice
            else:
                mask = cv2.imread(label_filename, cv2.IMREAD_GRAYSCALE)
        else:
            raise ValueError(f"Label file not found: {label_filename}")
        
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        
        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)
        mask = mask.astype('float32')
        
        return img, mask, {'img_id': img_id}


def sample_2d_slice_from_hwd_nnunet_style(volume_hwd, seg_hwd=None, oversample_foreground_percent=0.33, force_fg=None):
    if volume_hwd.ndim != 3:
        raise ValueError("volume_hwd must have shape (H,W,D)")

    d = volume_hwd.shape[2]
    seg_dhw = None
    if seg_hwd is not None:
        if seg_hwd.ndim != 3:
            raise ValueError("seg_hwd must have shape (H,W,D)")
        if seg_hwd.shape != volume_hwd.shape:
            raise ValueError("seg_hwd and volume_hwd must have the same shape")
        seg_dhw = np.transpose(seg_hwd, (2, 0, 1))

    if force_fg is None:
        force_fg = np.random.uniform() < float(oversample_foreground_percent)

    z_idx = None
    if force_fg and seg_dhw is not None:
        fg_voxels = np.argwhere(seg_dhw > 0)
        if len(fg_voxels) > 0:
            z_idx = int(fg_voxels[np.random.choice(len(fg_voxels)), 0])

    if z_idx is None:
        z_idx = int(np.random.randint(0, d))

    image_slice = volume_hwd[:, :, z_idx]
    seg_slice = seg_hwd[:, :, z_idx] if seg_hwd is not None else None
    return image_slice, seg_slice, z_idx, force_fg
