import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
from torchvision.utils import save_image

from monounet.mono_layer import Mono2DV3

__all__ = [
    'UNet',
    'MonoUNetBase',
    'MonoUNetE1',
    'MonoUNetE12',
    'MonoUNetE123',
    'MonoUNetE1234',
    'MonoUNetE12V2',
    'MonoUNetE123V2',
    'MonoUNetE1234V2',
    'MonoUNetE1Gated',
    'MonoUNetE123V2Gated',
    # 'MonoUNetE1234D1',
    ## Only run the next two if the previous one is successful.
    # 'MonoUNetE1234D12',
    # 'MonoUNetE1234D123',
    ## Run after experiments above with best configuration.
    # 'MonoUNetS2',
    # 'MonoUNetS4',
    ## Optional run to show saturation
    # 'MonoUNetS6',
]

# -------------------------
# Baseline Lean UNet
# -------------------------

class MonoUNetBase(nn.Module):
    def __init__(
        self,
        in_channels=1,
        num_classes=2,
        img_size=(256, 256),
        init_filters=2,
        max_filters=2,
        deep_supervision=True,
        encoder_cls=None,
        encoder_kwargs=None,
    ):
        super().__init__()

        num_stages = 7 if max(img_size[0], img_size[1]) <= 256 else 8
        filters = [min(max_filters, init_filters * 2**i) for i in range(num_stages)]
        encoder_kwargs = encoder_kwargs or {}

        if encoder_cls is None:
            self.encoder = XTinyEncoder(in_channels, filters, deep_supervision=deep_supervision)
        else:
            self.encoder = encoder_cls(in_channels, filters, deep_supervision=deep_supervision, **encoder_kwargs)

        self.decoder = XTinyDecoder(self.encoder, num_classes, filters, deep_supervision)
        self.initialize_weights()

    def forward(self, x):
        x, skip_connections = self.encoder(x)
        x = self.decoder(x, skip_connections)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    m.bias = nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm2d):
                if m.weight is not None:
                    m.weight = nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias = nn.init.constant_(m.bias, 0)


class XTinyEncoder(nn.Module):
    def __init__(self, in_channels=1, filters=None, deep_supervision=True):
        super().__init__()
        filters = filters or []

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, filters[0], kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(filters[0], eps=1e-5, momentum=0.1, affine=True, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

        stages = []
        pooling_layers = []

        num_stages = len(filters)
        for i in range(num_stages):
            block = nn.Sequential(
                nn.Conv2d(filters[i], filters[i], kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(filters[i], eps=1e-5, momentum=0.1, affine=True, track_running_stats=False),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
            )
            stages.append(nn.Sequential(block, block))

            if i < num_stages - 1:
                pooling_layers.append(nn.Conv2d(filters[i], filters[i + 1], kernel_size=3, stride=2, padding=1))

        self.stages = nn.ModuleList(stages)
        self.pooling_layers = nn.ModuleList(pooling_layers)

    def forward(self, x):
        x = self.stem(x)
        skip_connections = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.stages) - 1:
                skip_connections.append(x)
                x = self.pooling_layers[i](x)
        return x, skip_connections


class XTinyDecoder(nn.Module):
    def __init__(self, encoder, num_classes=2, filters=None, deep_supervision=True):
        super().__init__()
        filters = filters or []
        self.deep_supervision = deep_supervision

        stages = []
        transpose_convs = []
        ds_seg_heads = []

        num_stages = len(encoder.stages)
        for i in range(1, num_stages):
            stages.append(
                nn.Sequential(
                    nn.Conv2d(filters[-(i + 1)] * 2, filters[-(i + 1)], kernel_size=3, stride=1, padding=1),
                    nn.InstanceNorm2d(filters[-(i + 1)], eps=1e-5, momentum=0.1, affine=True, track_running_stats=False),
                    nn.LeakyReLU(negative_slope=0.01, inplace=True),
                )
            )
            transpose_convs.append(nn.ConvTranspose2d(filters[-i], filters[-(i + 1)], kernel_size=2, stride=2))
            ds_seg_heads.append(nn.Conv2d(filters[-(i + 1)], num_classes, kernel_size=1, stride=1))

        self.stages = nn.ModuleList(stages)
        self.transpose_convs = nn.ModuleList(transpose_convs)
        self.ds_seg_heads = nn.ModuleList(ds_seg_heads)

    def forward(self, x, skip_connections):
        outputs = []
        for i in range(len(self.stages)):
            x = self.transpose_convs[i](x)
            x = torch.cat([x, skip_connections[-(i + 1)]], dim=1)
            x = self.stages[i](x)
            outputs.append(self.ds_seg_heads[i](x))
        return outputs if self.deep_supervision else outputs[-1]


# -------------------------------------------
# Monogenic encoder with "inject up to stage"
# -------------------------------------------

class MonoUNetEEncoder(XTinyEncoder):
    """
    Inject monogenic features into encoder stages cumulatively:
    inject_upto = 1 -> stage 0 only
    inject_upto = 2 -> stages 0..1
    inject_upto = 3 -> stages 0..2
    inject_upto = 4 -> stages 0..3
    """
    def __init__(self, in_channels=1, filters=None, deep_supervision=True,
                 inject_upto: int = 1, gate_encoder: bool = True,
                 nscale: int | None = None, n_freq: int = 1):
        super().__init__(in_channels, filters, deep_supervision=deep_supervision)
        filters = filters or []
        num_stages = len(filters)

        assert 1 <= inject_upto <= num_stages, f"inject_upto must be in [1,{num_stages}]"
        self.inject_upto = inject_upto
        self.gate_encoder = gate_encoder

        self.mono2d = Mono2DV3(in_channels, nscale=nscale, n_freq=n_freq, norm="std", return_phase=True)

        # Build LP pyramid progressively (cheap + consistent)
        self.lp_down = nn.ModuleList([
            nn.AvgPool2d(kernel_size=2, stride=2)
            for _ in range(num_stages - 1)
        ])

        # Project monogenic features to match stage width (C_i)
        self.lp_proj = nn.ModuleList([
            nn.Conv2d(self.mono2d.out_channels, filters[i], kernel_size=1, stride=1, bias=False)
            for i in range(num_stages)
        ])

        if self.gate_encoder:
            # A very simple, stable residual gate per stage:
            # x <- x + sigma(a_i) * lp
            self.gate_a = nn.ParameterList([nn.Parameter(torch.zeros(filters[i])) for i in range(inject_upto)])

        # Replace pooling with your norm+act version (since your MonoUNetE1Encoder did)
        self.pooling_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(filters[i], filters[i+1], kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(filters[i+1], eps=1e-5, momentum=0.1, affine=True, track_running_stats=False),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
            )
            for i in range(num_stages - 1)
        ])

    def _inject(self, x, lp_stage, i: int):
        if not self.gate_encoder:
            return x + lp_stage
        alpha = torch.sigmoid(self.gate_a[i])  # scalar in (0,1)
        alpha = alpha.view(1, -1, 1, 1)
        return x + alpha * lp_stage

    def _forward_with_lp_feat(self, x, lp_feat):
        x = self.stem(x)          # B x C0 x H x W

        skip_connections = []
        for i, stage in enumerate(self.stages):
            if i < self.inject_upto:
                lp_stage = self.lp_proj[i](lp_feat)  # -> B x C_i x H_i x W_i
                x = self._inject(x, lp_stage, i)

            x = stage(x)

            if i < len(self.stages) - 1:
                skip_connections.append(x)
                x = self.pooling_layers[i](x)

                # progressive downsample of monogenic features to match next stage
                lp_feat = self.lp_down[i](lp_feat)

        return x, skip_connections

    def forward(self, x):
        lp_feat = self.mono2d(x)  # B x Cmono x H x W
        return self._forward_with_lp_feat(x, lp_feat)


class MonoUNetEEncoderV1(MonoUNetEEncoder):
    """
    V1 encoder that mixes monogenic features once at full resolution
    before projecting to stage widths.
    """
    def __init__(self, in_channels=1, filters=None, deep_supervision=True,
                 inject_upto: int = 1, gate_encoder: bool = True,
                 nscale: int | None = None, n_freq: int = 1):
        super().__init__(in_channels, filters, deep_supervision=deep_supervision,
                         inject_upto=inject_upto, gate_encoder=gate_encoder,
                         nscale=nscale, n_freq=n_freq)
        self.mono_mix = nn.Conv2d(self.mono2d.out_channels, self.mono2d.out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        lp_feat = self.mono2d(x)
        lp_feat = self.mono_mix(lp_feat)
        return self._forward_with_lp_feat(x, lp_feat)


# -------------------------
# MonoUNetE variants - Monogenic features injected into encoder stages
# -------------------------

class UNet(MonoUNetBase):
    def __init__(self, **kwargs):
        super().__init__(init_filters=32, max_filters=512, **kwargs)


class MonoUNetE1(MonoUNetBase):
    def __init__(self, **kwargs):
        super().__init__(encoder_cls=MonoUNetEEncoder, encoder_kwargs={"inject_upto": 1, "nscale": 3, "gate_encoder": False}, **kwargs)


class MonoUNetE1Gated(MonoUNetBase):
    def __init__(self, **kwargs):
        super().__init__(encoder_cls=MonoUNetEEncoderV1, encoder_kwargs={"inject_upto": 1, "nscale": 3, "gate_encoder": True}, **kwargs)


class MonoUNetE12(MonoUNetBase):
    def __init__(self, **kwargs):
        super().__init__(encoder_cls=MonoUNetEEncoder, encoder_kwargs={"inject_upto": 2, "nscale": 3, "gate_encoder": False}, **kwargs)

class MonoUNetE123(MonoUNetBase):
    def __init__(self, **kwargs):
        super().__init__(encoder_cls=MonoUNetEEncoder, encoder_kwargs={"inject_upto": 3, "nscale": 3, "gate_encoder": False}, **kwargs)

class MonoUNetE1234(MonoUNetBase):
    def __init__(self, **kwargs):
        super().__init__(encoder_cls=MonoUNetEEncoder, encoder_kwargs={"inject_upto": 4, "nscale": 3, "gate_encoder": False}, **kwargs)


class MonoUNetE12V2(MonoUNetBase):
    def __init__(self, **kwargs):
        super().__init__(encoder_cls=MonoUNetEEncoder, encoder_kwargs={"n_freq": 2, "inject_upto": 2, "nscale": 3, "gate_encoder": False}, **kwargs)


class MonoUNetE123V2(MonoUNetBase):
    def __init__(self, **kwargs):
        super().__init__(encoder_cls=MonoUNetEEncoder, encoder_kwargs={"n_freq": 3, "inject_upto": 3, "nscale": 3, "gate_encoder": False}, **kwargs)


class MonoUNetE123V2Gated(MonoUNetBase):
    def __init__(self, **kwargs):
        super().__init__(encoder_cls=MonoUNetEEncoderV1, encoder_kwargs={"n_freq": 3, "inject_upto": 3, "nscale": 3, "gate_encoder": True}, **kwargs)


class MonoUNetE1234V2(MonoUNetBase):
    def __init__(self, **kwargs):
        super().__init__(encoder_cls=MonoUNetEEncoder, encoder_kwargs={"n_freq": 4, "inject_upto": 4, "nscale": 3, "gate_encoder": False}, **kwargs)


class CascadeBase(nn.Module):
    def __init__(
        self,
        base_ckpt: str,
        base_arch: nn.Module,
        refiner_arch: MonoUNetBase,
        base_kwargs=None,
        refiner_kwargs=None,
        model_dir=None,
        debug_samples=2,
        mask_threshold=0.5,
        mask_class=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        base_kwargs = base_kwargs or {}
        refiner_kwargs = refiner_kwargs or {}

        self.base_model = base_arch(**base_kwargs)
        self.refiner_model = refiner_arch(**refiner_kwargs)
        self.model_dir = model_dir
        self.debug_samples = debug_samples
        self.mask_threshold = mask_threshold
        self.mask_class = mask_class

        # load base model
        ckpt = torch.load(base_ckpt, weights_only=False, map_location="cpu")
        state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
        self.base_model.load_state_dict(state)
        self.base_model.eval()
        for p in self.base_model.parameters():
            p.requires_grad = False

    def _base_to_mask(self, base_output):
        mask = torch.sigmoid(base_output)
        return (mask >= self.mask_threshold).float()

    def _apply_mask_augment(self, mask):
        elastic_prob = 0.5
        elastic_min = 50.0
        elastic_max = 250.0
        mask_patch_empty_prob = 0.02
        mask_patch_prob = 0.4
        mask_patch_bands = 4
        mask_patch_min_bands = 1
        mask_patch_max_bands = 3
        mask_foreground_prob = 0.25
        mask_foreground_blobs_min = 1
        mask_foreground_blobs_max = 1
        mask_foreground_radius_min = 6
        mask_foreground_radius_max = 24
        mask_dropout = 0.1
        mask_dropout_foreground_only = False

        if self.training and elastic_prob > 0:
            apply_elastic = (torch.rand(mask.size(0), device=mask.device) < elastic_prob)
            if apply_elastic.any():
                alpha = elastic_min + (elastic_max - elastic_min) * torch.rand(1, device=mask.device).item()
                elastic = v2.ElasticTransform(alpha=alpha)
                for idx in torch.nonzero(apply_elastic, as_tuple=False).flatten():
                    mask[idx] = elastic(mask[idx])
        if self.training and mask_patch_empty_prob > 0:
            empty_keep = (torch.rand(mask.size(0), 1, 1, 1, device=mask.device) >= mask_patch_empty_prob).float()
            mask = mask * empty_keep
        if self.training and mask_patch_prob > 0 and mask_patch_bands > 1:
            apply_patch = (torch.rand(mask.size(0), device=mask.device) < mask_patch_prob)
            if apply_patch.any():
                _, _, h, w = mask.shape
                bands = mask_patch_bands
                band_width = w // bands
                for idx in torch.nonzero(apply_patch, as_tuple=False).flatten():
                    k_min = max(1, min(mask_patch_min_bands, bands))
                    k_max = max(k_min, min(mask_patch_max_bands, bands))
                    k = int(torch.randint(k_min, k_max + 1, (1,), device=mask.device).item())
                    band_indices = torch.randperm(bands, device=mask.device)[:k]
                    for b in band_indices:
                        start = int(b.item() * band_width)
                        end = int((b.item() + 1) * band_width) if b.item() < bands - 1 else w
                        mask[idx, :, :, start:end] = 0
        if self.training and mask_foreground_prob > 0:
            apply_fg = (torch.rand(mask.size(0), device=mask.device) < mask_foreground_prob)
            if apply_fg.any():
                _, _, h, w = mask.shape
                for idx in torch.nonzero(apply_fg, as_tuple=False).flatten():
                    rmin = max(1, int(mask_foreground_radius_min))
                    rmax = max(rmin, int(mask_foreground_radius_max))
                    nmin = max(1, int(mask_foreground_blobs_min))
                    nmax = max(nmin, int(mask_foreground_blobs_max))
                    n = int(torch.randint(nmin, nmax + 1, (1,), device=mask.device).item())
                    for _ in range(n):
                        k = int(torch.randint(rmin, rmax + 1, (1,), device=mask.device).item())
                        k = max(3, k | 1)
                        k = min(k, h if h % 2 == 1 else h - 1, w if w % 2 == 1 else w - 1)
                        noise = torch.rand(1, 1, h, w, device=mask.device)
                        smooth = F.avg_pool2d(noise, kernel_size=k, stride=1, padding=k // 2)
                        q = 0.97 + 0.02 * torch.rand(1, device=mask.device)
                        thresh = torch.quantile(smooth.view(-1), q)
                        blob = (smooth > thresh).float()
                        blob = F.max_pool2d(blob, kernel_size=3, stride=1, padding=1)
                        mask[idx, 0] = torch.clamp(mask[idx, 0] + blob[0, 0], 0, 1)
        if self.training and mask_dropout > 0:
            if mask_dropout_foreground_only:
                keep = (torch.rand_like(mask) >= mask_dropout).float()
                mask = mask * keep
            else:
                keep = (torch.rand(mask.size(0), 1, 1, 1, device=mask.device) >= mask_dropout).float()
                mask = mask * keep
        return mask

    def _save_debug_refiner(self, input_mask, output):
        debug_dir = os.path.join(self.model_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)
        if isinstance(output, (list, tuple)):
            output = output[-1]
        count = min(self.debug_samples, output.size(0))
        probs = torch.sigmoid(output)
        masks = (probs >= self.mask_threshold).float()
        for i in range(count):
            gap = torch.ones_like(input_mask[i][:, :, :2])
            combo = torch.cat([input_mask[i], gap, masks[i]], dim=2)
            save_image(combo, os.path.join(debug_dir, f"sample_{i}_refiner.png"))

    def forward(self, x):
        with torch.no_grad():
            base_output = self.base_model(x)
        mask = self._base_to_mask(base_output)
        if self.training:
            mask = self._apply_mask_augment(mask)
        refined_input = torch.cat([x, mask], dim=1)
        output = self.refiner_model(refined_input)
        if self.training:
            self._save_debug_refiner(mask, output)
        return output
