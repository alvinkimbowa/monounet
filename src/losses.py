import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

__all__ = ['BCEDiceLoss', 'CEDiceLoss', 'LovaszHingeLoss', 'TinyUNetLoss', 'UNetLoss']


def softmax_helper_dim1(x):
    return F.softmax(x, dim=1)


class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    def forward(self, input, target):
        if target.dtype != torch.long:
            target = target.long()
        return super().forward(input, target)


def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    if axes is None:
        axes = tuple(range(2, net_output.ndim))

    with torch.no_grad():
        if gt.ndim == net_output.ndim:
            if gt.shape[1] == 1:
                gt = gt[:, 0]
            else:
                gt = torch.argmax(gt, dim=1)
        gt = gt.long()
        gt_onehot = F.one_hot(gt, num_classes=net_output.shape[1]).permute(0, -1, *range(1, gt.ndim)).float()

    if mask is not None:
        mask = mask.float()
        if mask.ndim == net_output.ndim - 1:
            mask = mask.unsqueeze(1)
        net_output = net_output * mask
        gt_onehot = gt_onehot * mask

    tp = net_output * gt_onehot
    fp = net_output * (1 - gt_onehot)
    fn = (1 - net_output) * gt_onehot
    tn = (1 - net_output) * (1 - gt_onehot)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    tp = tp.sum(dim=axes)
    fp = fp.sum(dim=axes)
    fn = fn.sum(dim=axes)
    tn = tn.sum(dim=axes)
    return tp, fp, fn, tn


class SoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.):
        super().__init__()
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape
        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)
        nominator = 2 * tp
        denominator = 2 * tp + fp + fn
        dc = (nominator + self.smooth) / torch.clamp(denominator + self.smooth, min=1e-8)

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()
        return -dc


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        if target.ndim == input.ndim - 1:
            target = target.unsqueeze(1)
        elif target.ndim == input.ndim and target.shape[1] != input.shape[1]:
            target = target[:, :1]
        target = target.float()

        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return bce + dice


class UNetLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        ce = nn.CrossEntropyLoss()
        return ce(input, target.squeeze(1).long())

class TinyUNetLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        ce = nn.CrossEntropyLoss()

        # Accept class-index masks [B,H,W] / [B,1,H,W] and one-hot [B,C,H,W].
        if target.ndim == input.ndim and target.shape[1] > 1:
            target = torch.argmax(target, dim=1)
        elif target.ndim == input.ndim and target.shape[1] == 1:
            target = target[:, 0]
        target = target.long().clamp(0, input.shape[1] - 1)

        smooth = 1e-5
        input_soft = F.softmax(input, dim=1)
        target_oh = F.one_hot(target, num_classes=input.shape[1]).permute(0, 3, 1, 2).float()

        # For multi-class Dice in this loss, exclude background channel 0.
        if input.shape[1] > 1:
            input_soft = input_soft[:, 1:, :, :]
            target_oh = target_oh[:, 1:, :, :]

        num = target_oh.size(0)
        input_soft = input_soft.reshape(num, -1)
        target_oh = target_oh.reshape(num, -1)
        intersection = (input_soft * target_oh)
        dice = (2. * intersection.sum(1) + smooth) / (input_soft.sum(1) + target_oh.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return ce(input, target) + dice


class CEDiceLoss(nn.Module):
    def __init__(self, soft_dice_kwargs=None, ce_kwargs=None, weight_ce=1, weight_dice=1, ignore_label=None):
        super().__init__()
        if soft_dice_kwargs is None:
            soft_dice_kwargs = {'batch_dice': False, 'do_bg': False, 'smooth': 1.}
        if ce_kwargs is None:
            ce_kwargs = {}
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output, target):
        if target.ndim == net_output.ndim and target.shape[1] > 1:
            target = torch.argmax(target, dim=1, keepdim=True)
        elif target.ndim == net_output.ndim - 1:
            target = target.unsqueeze(1)

        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables'
            mask = target != self.ignore_label
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0
        return self.weight_ce * ce_loss + self.weight_dice * dc_loss


class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss
