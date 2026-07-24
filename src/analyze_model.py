#!/usr/bin/env python3
"""
Analyze model parameters and FLOPs
Callable from command line or from run_train.sh
"""
import argparse
import json
import os
import torch
import torch.nn as nn
from os.path import join

from thop import profile, clever_format

import archs
from TinyUNet import TinyUNet
from monounet import MonogenicNets, MonoUNets
from utils import str2bool

ARCH_NAMES = archs.__all__
MONO_ARCH_NAMES = MonogenicNets.__all__
MONOUNET_ARCH_NAMES = MonoUNets.__all__


def load_model(arch, input_channels, num_classes, input_h, input_w, deep_supervision=False):
    """Load model the same way train.py does"""
    if arch == "UNext" or arch == "UNext_S" or arch == "CMUNeXt-S":
        model = archs.__dict__[arch](num_classes, input_channels, deep_supervision)
    elif arch == "TinyUNet":
        model = TinyUNet(input_channels, num_classes)
    elif arch in MONO_ARCH_NAMES:
        model = MonogenicNets.__dict__[arch](
            input_channels,
            num_classes,
            img_size=(input_h, input_w),
            deep_supervision=deep_supervision
        )
    elif arch in MONOUNET_ARCH_NAMES:
        model = MonoUNets.__dict__[arch](
            in_channels=input_channels,
            num_classes=num_classes,
            img_size=(input_h, input_w),
            deep_supervision=deep_supervision
        )
    else:
        raise NotImplementedError(f"Architecture {arch} not supported")
    
    return model


def count_parameters(model):
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def count_flops(model, input_tensor):
    """Count FLOPs using thop"""
    macs, params = profile(model, inputs=(input_tensor,), verbose=False)
    flops = macs * 2  # MACs to FLOPS (multiply-add operations)
    return macs, flops, params


def analyze_model(arch, input_channels=1, num_classes=2, input_h=256, input_w=256, 
                  deep_supervision=False, gpu=0, save_path=None):
    """Analyze model parameters and FLOPs"""
    
    # Set device
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*60}")
    print(f"MODEL ANALYSIS")
    print(f"{'='*60}")
    print(f"Architecture: {arch}")
    print(f"Input channels: {input_channels}")
    print(f"Number of classes: {num_classes}")
    print(f"Input size: {input_h}x{input_w}")
    print(f"Deep supervision: {deep_supervision}")
    print(f"Device: {device}")
    print(f"{'='*60}")
    
    # Load model
    model = load_model(arch, input_channels, num_classes, input_h, input_w, deep_supervision)
    model = model.to(device)
    model.eval()
    print("✓ Model loaded successfully")
    
    # Create input tensor
    input_tensor = torch.randn(1, input_channels, input_h, input_w).to(device)
    
    # Count parameters
    total_params, trainable_params = count_parameters(model)
    
    # Count FLOPs
    macs, flops, thop_params = count_flops(model, input_tensor)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"PARAMETERS:")
    print(f"  Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print(f"  Non-trainable parameters: {total_params - trainable_params:,}")
    
    print(f"\nCOMPUTATIONAL COMPLEXITY:")
    print(f"  MACs (Multiply-Accumulates): {macs:,} ({macs/1e9:.2f}G)")
    print(f"  FLOPS (Floating Point Operations): {flops:,} ({flops/1e9:.2f}G)")
    print(f"  thop parameter count: {thop_params:,}")
    
    print(f"{'='*60}\n")
    
    # Prepare metrics dictionary
    metrics = {
        "architecture": arch,
        "input_channels": input_channels,
        "num_classes": num_classes,
        "input_size": [input_h, input_w],
        "deep_supervision": deep_supervision,
        "device": str(device),
        "parameters": {
            "total": int(total_params),
            "trainable": int(trainable_params),
            "non_trainable": int(total_params - trainable_params),
            "total_millions": round(total_params/1e6, 2),
            "trainable_millions": round(trainable_params/1e6, 2)
        }
    }
    
    metrics["macs"] = {
        "total": int(macs),
        "giga": round(macs/1e9, 2)
    }
    metrics["flops"] = {
        "total": int(flops),
        "giga": round(flops/1e9, 2)
    }
    
    # Save metrics to JSON
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {save_path}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Analyze model parameters and FLOPs')
    parser.add_argument('--arch', '-a', type=str, required=True,
                        help='Model architecture name')
    parser.add_argument('--input_channels', type=int, default=1,
                        help='Number of input channels (default: 1)')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of output classes (default: 2)')
    parser.add_argument('--input_h', type=int, default=256,
                        help='Input height (default: 256)')
    parser.add_argument('--input_w', type=int, default=256,
                        help='Input width (default: 256)')
    parser.add_argument('--deep_supervision', default=False, type=str2bool,
                        help='Use deep supervision')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID (default: 0, use -1 for CPU)')
    parser.add_argument('--save', type=str, default=None,
                        help='Path to save metrics JSON (optional)')
    
    args = parser.parse_args()
    print("save path: ", args.save)
    analyze_model(
        arch=args.arch,
        input_channels=args.input_channels,
        num_classes=args.num_classes,
        input_h=args.input_h,
        input_w=args.input_w,
        deep_supervision=args.deep_supervision,
        gpu=args.gpu,
        save_path=args.save
    )


if __name__ == '__main__':
    main()
