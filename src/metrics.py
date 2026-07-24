import torch
import torch.nn.functional as F
from monai.metrics import DiceMetric, MeanIoU


def _to_label_map(output, target, num_classes):
    # output: logits [B,C,H,W] or label map [B,H,W]
    if output.ndim == 4:
        if num_classes == 1:
            if output.shape[1] == 1:
                output = (torch.sigmoid(output[:, 0]) >= 0.5).long()
            else:
                output = torch.argmax(output, dim=1)
        else:
            output = torch.argmax(output, dim=1)
    output = output.long()

    # target: [B,H,W] or [B,1,H,W] or one-hot [B,C,H,W]
    if target.ndim == 4 and target.shape[1] == num_classes:
        target = torch.argmax(target, dim=1)
    elif target.ndim == 4 and target.shape[1] == 1:
        target = target[:, 0]
    target = target.long()

    return output, target


def _to_one_hot(label_map, num_classes):
    return F.one_hot(label_map, num_classes=num_classes).permute(0, 3, 1, 2).float()


def iou_dice_metrics(output, target, num_classes, include_background=False):
    pred_lbl, tgt_lbl = _to_label_map(output, target, num_classes)

    if num_classes == 1:
        y_pred = pred_lbl.unsqueeze(1).float()
        y = tgt_lbl.unsqueeze(1).float()
    else:
        y_pred = _to_one_hot(pred_lbl, num_classes)
        y = _to_one_hot(tgt_lbl, num_classes)

    iou_metric = MeanIoU(include_background=include_background, reduction='mean', get_not_nans=False)
    dice_metric = DiceMetric(include_background=include_background, reduction='mean', get_not_nans=False)

    iou = torch.nanmean(iou_metric(y_pred=y_pred, y=y)).item()
    dice = torch.nanmean(dice_metric(y_pred=y_pred, y=y)).item()
    return iou, dice


def iou_score(output, target, num_classes, include_background=False):
    iou, _ = iou_dice_metrics(output, target, num_classes, include_background=include_background)
    return iou


def dice_coef(output, target, num_classes, include_background=False):
    _, dice = iou_dice_metrics(output, target, num_classes, include_background=include_background)
    return dice
