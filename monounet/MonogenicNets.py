import torch
import torch.nn as nn

from monounet.mono_layer import Mono2D, Mono2DV2

__all__ = [
    'UNetBaseline',
    'XTinyUNet',
    'XTinyUNetB',
    'XTinyUNetL',
    'XTinyUNetH',
    'XTinyUNetXL',
    'XTinynnUNet',
    'XTinyMonoUNetScale1',
    'XTinyMonoUNetScale6',
    'XTinyMonoV2UNetScale1',
    'XTinyMonoV2UNetScale6',
    'XTinyMonoV2GatedUNet',
    'XTinyMonoV2GatedEncUNet',
    'XTinyMonoV2GatedEncUNetV1',
    'XTinyMonoV2GatedEncUNetV1B',
    'XTinyMonoV2GatedEncUNetV1L',
    'XTinyMonoV2GatedEncUNetV1H',
    'XTinyMonoV2GatedEncUNetV1XL',
    'XTinyMonoV2GatedEncUNetV0',
    'XTinyMonoV2GatedEncDecUNet',
    'XTinyMonoV2GatedEncDecUNetV1',
    'XTinyMonoV2GatedDecUNet',
    'XTinyMonoV2GatedDecUNetV1'
]


def center_crop(skip, target_size):
    """Center crop skip connection to match target size."""
    skip_h, skip_w = skip.shape[-2:]
    target_h, target_w = target_size
    offset_h = (skip_h - target_h) // 2
    offset_w = (skip_w - target_w) // 2
    cropped = skip[..., offset_h:offset_h + target_h, offset_w:offset_w + target_w]
    return cropped


class XTinyUNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, img_size=(256, 256), init_filters=1, max_filters=2, deep_supervision=True):
        super().__init__()
        
        num_stages = 7 if max(img_size[0], img_size[1]) <= 256 else 8
        filters = [min(max_filters, init_filters * 2**i) for i in range(num_stages)]
        
        self.encoder = XTinyEncoder(in_channels, filters, deep_supervision=deep_supervision)
        self.decoder = XTinyDecoder(self.encoder, num_classes, filters, deep_supervision)

        self.initialize_weights()

    def forward(self, x):
        x, skip_connections = self.encoder(x)
        x = self.decoder(x, skip_connections)
        return x
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    m.bias = nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight = nn.init.constant_(m.weight, 1)
                m.bias = nn.init.constant_(m.bias, 0)

class UNetBaseline(XTinyUNet):
    def __init__(self, in_channels=1, num_classes=2, img_size=(256, 256), init_filters=32, max_filters=512, deep_supervision=True):
        super().__init__(in_channels, num_classes, img_size, init_filters, max_filters, deep_supervision)

        num_stages = 7 if max(img_size[0], img_size[1]) <= 256 else 8
        filters = [min(max_filters, init_filters * 2**i) for i in range(num_stages)]
        
        self.encoder = XTinyEncoder(in_channels, filters, deep_supervision=deep_supervision)
        self.decoder = UNetDecoder(self.encoder, num_classes, filters, deep_supervision)

        self.initialize_weights()


class XTinyUNetB(XTinyUNet):
    def __init__(self, in_channels=1, num_classes=2, img_size=(256, 256), init_filters=1, max_filters=4, deep_supervision=True):
        max_filters = 4
        super().__init__(in_channels, num_classes, img_size, init_filters, max_filters, deep_supervision)


class XTinyUNetL(XTinyUNet):
    def __init__(self, in_channels=1, num_classes=2, img_size=(256, 256), init_filters=1, max_filters=4, deep_supervision=True):
        max_filters = 8
        super().__init__(in_channels, num_classes, img_size, init_filters, max_filters, deep_supervision)


class XTinyUNetH(XTinyUNet):
    def __init__(self, in_channels=1, num_classes=2, img_size=(256, 256), init_filters=1, max_filters=4, deep_supervision=True):
        max_filters = 16
        super().__init__(in_channels, num_classes, img_size, init_filters, max_filters, deep_supervision)


class XTinyUNetXL(XTinyUNet):

    def __init__(self, in_channels=1, num_classes=2, img_size=(256, 256), init_filters=1, max_filters=4, deep_supervision=True):
        max_filters = 32
        super().__init__(in_channels, num_classes, img_size, init_filters, max_filters, deep_supervision)


class XTinynnUNet(XTinyUNet):
    def __init__(self, in_channels=1, num_classes=2, img_size=(256, 256), init_filters=32, max_filters=512, deep_supervision=True):
        super().__init__(in_channels, num_classes, img_size, init_filters, max_filters, deep_supervision)


class XTinyEncoder(nn.Module):
    def __init__(self, in_channels=1, filters=[], deep_supervision=True):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, filters[0], kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(filters[0], eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )

        stages = []
        pooling_layers = []

        num_stages = len(filters)
        for i in range(num_stages):            
            # Conv Block
            block = nn.Sequential(
                nn.Conv2d(filters[i], filters[i], kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(filters[i], eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
                nn.LeakyReLU(negative_slope=0.01, inplace=True)
            )
            stages.append(nn.Sequential(block, block))

            if i < num_stages - 1:
                # Pooling Block
                pooling_layers.append(nn.Conv2d(filters[i], filters[i+1], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)))
        
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
    def __init__(self, encoder, num_classes=2, filters=[], deep_supervision=True):
        super().__init__()
        
        self.deep_supervision = deep_supervision
        stages = []
        num_stages = len(encoder.stages)
        transpose_convs = []
        ds_seg_heads = []
        for i in range(1, num_stages):
            stages.append(nn.Sequential(
                nn.Conv2d(filters[-(i+1)]*2, filters[-(i+1)], kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(filters[-(i+1)], eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
                nn.LeakyReLU(negative_slope=0.01, inplace=True)
            ))

            transpose_convs.append(nn.ConvTranspose2d(filters[-i], filters[-(i+1)], kernel_size=2, stride=2))
            
            ds_seg_heads.append(nn.Sequential(
                nn.Conv2d(filters[-(i+1)], num_classes, kernel_size=1, stride=1)
            ))
        
        self.stages = nn.ModuleList(stages)
        self.transpose_convs = nn.ModuleList(transpose_convs)
        self.ds_seg_heads = nn.ModuleList(ds_seg_heads)

    def forward(self, x, skip_connections):
        outputs = []
        for i in range(len(self.stages)):
            x = self.transpose_convs[i](x)
            x = torch.cat([x, skip_connections[-(i+1)]], dim=1)
            x = self.stages[i](x)
            outputs.append(self.ds_seg_heads[i](x))
        
        if self.deep_supervision:
            return outputs
        else:
            return outputs[-1]


class UNetDecoder(XTinyDecoder):
    def __init__(self, encoder, num_classes=2, filters=[], deep_supervision=True):
        super().__init__(encoder, num_classes, filters, deep_supervision)
        
        self.deep_supervision = deep_supervision
        stages = []
        num_stages = len(encoder.stages)
        transpose_convs = []
        ds_seg_heads = []
        for i in range(1, num_stages):
            stages.append(nn.Sequential(
                nn.Conv2d(filters[-(i+1)]*2, filters[-(i+1)], kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(filters[-(i+1)], eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.Conv2d(filters[-(i+1)], filters[-(i+1)], kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(filters[-(i+1)], eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
                nn.LeakyReLU(negative_slope=0.01, inplace=True)
            ))

            transpose_convs.append(nn.ConvTranspose2d(filters[-i], filters[-(i+1)], kernel_size=2, stride=2))
            
            ds_seg_heads.append(nn.Sequential(
                nn.Conv2d(filters[-(i+1)], num_classes, kernel_size=1, stride=1)
            ))
        
        self.stages = nn.ModuleList(stages)
        self.transpose_convs = nn.ModuleList(transpose_convs)
        self.ds_seg_heads = nn.ModuleList(ds_seg_heads)


class XTinyMonoUNetScale1(XTinyUNet):
    def __init__(self, in_channels=1, num_classes=2, img_size=(256, 256), init_filters=1, max_filters=2, deep_supervision=True):
        super().__init__(in_channels, num_classes, img_size, init_filters, max_filters, deep_supervision)
        self.mono2d = Mono2D(in_channels, nscale=1, norm="std", return_phase=True)
        
    def forward(self, x):
        x = self.mono2d(x)
        return super().forward(x)


class XTinyMonoUNetScale6(XTinyMonoUNetScale1):
    def __init__(self, in_channels=1, num_classes=2, img_size=(256, 256), init_filters=1, max_filters=2, deep_supervision=True):
        super().__init__(in_channels, num_classes, img_size, init_filters, max_filters, deep_supervision)
        self.mono2d = Mono2D(in_channels, nscale=6, norm="std", return_phase=True)


class XTinyMonoV2UNetScale1(XTinyMonoUNetScale1):
    def __init__(self, in_channels=1, num_classes=2, img_size=(256, 256), init_filters=1, max_filters=2, deep_supervision=True):
        super().__init__(in_channels, num_classes, img_size, init_filters, max_filters, deep_supervision)
        print("in_channels", in_channels)
        self.mono2d = Mono2DV2(in_channels, nscale=1, norm="std", return_phase=True)
        in_channels = self.mono2d.out_channels
        print("in_channels after Mono2DV2", in_channels)
        
        num_stages = 7 if max(img_size[0], img_size[1]) <= 256 else 8
        filters = [min(max_filters, init_filters * 2**i) for i in range(num_stages)]
        
        self.encoder = XTinyEncoder(in_channels, filters, deep_supervision=deep_supervision)
        self.decoder = XTinyDecoder(self.encoder, num_classes, filters, deep_supervision)

        self.initialize_weights()


class XTinyMonoV2UNetScale6(XTinyMonoUNetScale1):
    def __init__(self, in_channels=1, num_classes=2, img_size=(256, 256), init_filters=1, max_filters=2, deep_supervision=True):
        super().__init__(in_channels, num_classes, img_size, init_filters, max_filters, deep_supervision)
        print("in_channels", in_channels)
        self.mono2d = Mono2DV2(in_channels, nscale=6, norm="std", return_phase=True)
        in_channels = self.mono2d.out_channels
        print("in_channels after Mono2DV2", in_channels)
        
        num_stages = 7 if max(img_size[0], img_size[1]) <= 256 else 8
        filters = [min(max_filters, init_filters * 2**i) for i in range(num_stages)]
        
        self.encoder = XTinyEncoder(in_channels, filters, deep_supervision=deep_supervision)
        self.decoder = XTinyDecoder(self.encoder, num_classes, filters, deep_supervision)

        self.initialize_weights()


class XTinyMonoV2GatedUNet(XTinyUNet):
    def __init__(self, in_channels=1, num_classes=2, img_size=(256, 256), init_filters=1, max_filters=2, deep_supervision=True):
        super().__init__(in_channels, num_classes, img_size, init_filters, max_filters, deep_supervision)
        self.mono2d = Mono2DV2(in_channels, nscale=1, norm="std", return_phase=True)
        
        # Learnable affine for gate and residual strength
        self.gate_weight = nn.Parameter(torch.ones(1))
        self.gate_bias = nn.Parameter(torch.zeros(1))
        self.alpha = nn.Parameter(torch.randn(1))
        print("\n\nUsing MonoGatedUNet with mono2d layer\n\n")
    
    def forward(self, x):
        mono_feat = self.mono2d(x)
        gate = torch.sigmoid(self.gate_weight * mono_feat + self.gate_bias)
        x = x * (1 + self.alpha * gate)     # Use a residual connection to stabilize training
        return super().forward(x)


class XTinyMonoV2GatedEncUNet(XTinyUNet):
    def __init__(self, in_channels=1, num_classes=2, img_size=(256, 256), init_filters=1, max_filters=2, deep_supervision=True):
        super().__init__(in_channels, num_classes, img_size, init_filters, max_filters, deep_supervision)
        
        num_stages = 7 if max(img_size[0], img_size[1]) <= 256 else 8
        filters = [min(max_filters, init_filters * 2**i) for i in range(num_stages)]
        
        self.encoder = XTinyGatedEncoder(in_channels, filters, deep_supervision=deep_supervision)
        self.decoder = XTinyDecoder(self.encoder, num_classes, filters, deep_supervision)

        self.initialize_weights()


class XTinyMonoV2GatedEncUNetV1(XTinyUNet):
    def __init__(self, in_channels=1, num_classes=2, img_size=(256, 256), init_filters=1, max_filters=2, deep_supervision=True):
        super().__init__(in_channels, num_classes, img_size, init_filters, max_filters, deep_supervision)
        
        num_stages = 7 if max(img_size[0], img_size[1]) <= 256 else 8
        filters = [min(max_filters, init_filters * 2**i) for i in range(num_stages)]
        
        self.encoder = XTinyGatedEncoderV1(in_channels, filters, deep_supervision=deep_supervision)
        self.decoder = XTinyDecoder(self.encoder, num_classes, filters, deep_supervision)

        self.initialize_weights()


class XTinyMonoV2GatedEncUNetV1B(XTinyMonoV2GatedEncUNetV1):
    def __init__(self, in_channels=1, num_classes=2, img_size=(256, 256), init_filters=1, max_filters=2, deep_supervision=True):
        max_filters = 4
        super().__init__(in_channels, num_classes, img_size, init_filters, max_filters, deep_supervision)


class XTinyMonoV2GatedEncUNetV1L(XTinyMonoV2GatedEncUNetV1):
    def __init__(self, in_channels=1, num_classes=2, img_size=(256, 256), init_filters=1, max_filters=2, deep_supervision=True):
        max_filters = 8
        super().__init__(in_channels, num_classes, img_size, init_filters, max_filters, deep_supervision)


class XTinyMonoV2GatedEncUNetV1H(XTinyMonoV2GatedEncUNetV1):
    def __init__(self, in_channels=1, num_classes=2, img_size=(256, 256), init_filters=1, max_filters=2, deep_supervision=True):
        max_filters = 16
        super().__init__(in_channels, num_classes, img_size, init_filters, max_filters, deep_supervision)


class XTinyMonoV2GatedEncUNetV1XL(XTinyMonoV2GatedEncUNetV1):
    def __init__(self, in_channels=1, num_classes=2, img_size=(256, 256), init_filters=1, max_filters=2, deep_supervision=True):
        max_filters = 32
        super().__init__(in_channels, num_classes, img_size, init_filters, max_filters, deep_supervision)


class XTinyGatedEncoder(XTinyEncoder):
    def __init__(self, in_channels=1, filters=[], deep_supervision=True):
        super().__init__(in_channels, filters, deep_supervision=deep_supervision)
        
        num_stages = len(filters)
        self.mono2d = Mono2DV2(in_channels, nscale=num_stages + 1, norm="std", return_phase=True)

        self.lp_conv_layers = nn.ModuleList(
            [nn.Conv2d(self.mono2d.out_channels, self.mono2d.out_channels, kernel_size=3, stride=2, padding=1, bias=False) for i in range(num_stages - 1)]
        )
        self.lp_proj_layers = nn.ModuleList([nn.Conv2d(self.mono2d.out_channels, filters[i], kernel_size=1, stride=1, bias=False) for i in range(num_stages)])
        self.gate_weights = nn.ParameterList([nn.Parameter(torch.ones(1)) for i in range(len(self.lp_conv_layers) + 1)])
        self.gate_biases = nn.ParameterList([nn.Parameter(torch.zeros(1)) for i in range(len(self.lp_conv_layers) + 1)])
        self.alphas = nn.ParameterList([nn.Parameter(torch.randn(1)) for i in range(len(self.lp_conv_layers) + 1)])

        self.pooling_layers = nn.ModuleList([nn.Sequential(
                    nn.Conv2d(filters[i], filters[i+1], kernel_size=3, stride=2, padding=1),
                    nn.InstanceNorm2d(filters[i+1], eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
                    nn.LeakyReLU(negative_slope=0.01, inplace=True)
                ) for i in range(num_stages - 1)])

        self.return_lp = False
        self.gate_encoder = True

    def forward(self, x):
        x = self.stem(x)
        lp_feat = self.mono2d(x)
        
        lp_skips = []
        skip_connections = []
        for i, stage in enumerate(self.stages):
            lp_stage = self.lp_proj_layers[i](lp_feat)
            if self.gate_encoder:
                x = self.apply_gate(x, lp_stage, i)
            x = stage(x)
            if i < len(self.stages) - 1:
                skip_connections.append(x)
                lp_skips.append(lp_feat)
                x = self.pooling_layers[i](x)
                lp_feat = self.lp_conv_layers[i](lp_feat)
        
        if self.return_lp:
            return x, skip_connections, lp_skips
        else:
            return x, skip_connections
    
    def apply_gate(self, x: torch.Tensor, lp_features: torch.Tensor, layer_idx: int) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_weights[layer_idx] * lp_features + self.gate_biases[layer_idx])
        return x * (1 + self.alphas[layer_idx] * gate)


class XTinyGatedEncoderV1(XTinyGatedEncoder):
    def __init__(self, in_channels=1, filters=[], deep_supervision=True):
        super().__init__(in_channels, filters, deep_supervision=deep_supervision)
        
        self.gate_weights = nn.ParameterList([nn.Parameter(torch.ones(filters[i])) for i in range(len(filters))])
        self.gate_biases = nn.ParameterList([nn.Parameter(torch.zeros(filters[i])) for i in range(len(filters))])
        self.alphas = nn.ParameterList([nn.Parameter(torch.randn(filters[i])) for i in range(len(filters))])
    
    def apply_gate(self, x: torch.Tensor, lp_features: torch.Tensor, layer_idx: int) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_weights[layer_idx].view(1, -1, 1, 1) * lp_features + self.gate_biases[layer_idx].view(1, -1, 1, 1))
        return x * (1 + self.alphas[layer_idx].view(1, -1, 1, 1) * gate)



class XTinyMonoV2GatedEncDecUNet(XTinyMonoV2GatedEncUNet):
    def __init__(self, in_channels=1, num_classes=2, img_size=(256, 256), init_filters=1, max_filters=2, deep_supervision=True):
        super().__init__(in_channels, num_classes, img_size, init_filters, max_filters, deep_supervision)
        
        self.encoder.return_lp = True

        num_stages = 7 if max(img_size[0], img_size[1]) <= 256 else 8
        filters = [min(max_filters, init_filters * 2**i) for i in range(num_stages)]

        self.decoder = XTinyGatedDecoder(self.encoder, num_classes, filters, deep_supervision)

        self.initialize_weights()
    
    def forward(self, x):
        x, skip_connections, lp_skips = self.encoder(x)
        x = self.decoder(x, skip_connections, lp_skips)
        return x

class XTinyGatedDecoder(XTinyDecoder):
    def __init__(self, encoder, num_classes=2, filters=[], deep_supervision=True):
        super().__init__(encoder, num_classes, filters, deep_supervision)
        
        stages = []
        num_stages = len(encoder.stages)
        for i in range(1, num_stages):
            stages.append(nn.Sequential(
                nn.Conv2d(filters[-(i+1)]*3, filters[-(i+1)], kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(filters[-(i+1)], eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
                nn.LeakyReLU(negative_slope=0.01, inplace=True)
            ))
        
        self.stages = nn.ModuleList(stages)
        self.lp_proj_layers = nn.ModuleList([nn.Conv2d(encoder.mono2d.out_channels, filters[-(i+1)], kernel_size=1, stride=1, bias=False) for i in range(1, num_stages)])
        self.gate_weights = nn.ParameterList([nn.Parameter(torch.ones(1)) for i in range(len(encoder.stages))])
        self.gate_biases = nn.ParameterList([nn.Parameter(torch.zeros(1)) for i in range(len(encoder.stages))])
        self.alphas = nn.ParameterList([nn.Parameter(torch.randn(1)) for i in range(len(encoder.stages))])
        
    def forward(self, x, skip_connections, lp_skips):
        outputs = []
        for i in range(len(self.stages)):
            x = self.transpose_convs[i](x)
            lp_stage = self.lp_proj_layers[i](lp_skips[-(i+1)])
            x = torch.cat([x, skip_connections[-(i+1)], lp_stage], dim=1)
            x = self.stages[i](x)
            outputs.append(self.ds_seg_heads[i](x))
        
        if self.deep_supervision:
            return outputs
        else:
            return outputs[-1]


class XTinyMonoV2GatedEncUNetV0(XTinyUNet):
    def __init__(self, in_channels=1, num_classes=2, img_size=(256, 256), init_filters=1, max_filters=2, deep_supervision=True):
        super().__init__(in_channels, num_classes, img_size, init_filters, max_filters, deep_supervision)
        
        num_stages = 7 if max(img_size[0], img_size[1]) <= 256 else 8
        filters = [min(max_filters, init_filters * 2**i) for i in range(num_stages)]
        
        self.encoder = XTinyGatedEncoderV0(in_channels, filters, deep_supervision=deep_supervision)
        self.decoder = XTinyDecoder(self.encoder, num_classes, filters, deep_supervision)

        self.initialize_weights()


class XTinyGatedEncoderV0(XTinyGatedEncoder):
    def __init__(self, in_channels=1, filters=[], deep_supervision=True):
        super().__init__(in_channels, filters, deep_supervision=deep_supervision)
        
        num_stages = len(filters)
        self.mono2d = Mono2DV2(in_channels, nscale=num_stages + 1, norm="std", return_phase=True)

        self.lp_proj_layers = nn.ModuleList([nn.Conv2d(self.mono2d.out_channels, filters[i], kernel_size=1, stride=1, bias=False) for i in range(num_stages)])
        self.gate_weights = nn.ParameterList([nn.Parameter(torch.ones(1)) for i in range(num_stages)])
        self.gate_biases = nn.ParameterList([nn.Parameter(torch.zeros(1)) for i in range(num_stages)])
        self.alphas = nn.ParameterList([nn.Parameter(torch.randn(1)) for i in range(num_stages)])

        self.pooling_layers = nn.ModuleList([nn.Sequential(
                    nn.Conv2d(filters[i], filters[i+1], kernel_size=3, stride=2, padding=1),
                    nn.InstanceNorm2d(filters[i+1], eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
                    nn.LeakyReLU(negative_slope=0.01, inplace=True)
                ) for i in range(num_stages - 1)])

        self.return_lp = False
        self.gate_encoder = True

    def forward(self, x):
        x = self.stem(x)
        lp_feat = self.mono2d(x)
        
        lp_skips = []
        skip_connections = []
        for i, stage in enumerate(self.stages):
            lp_stage = self.lp_proj_layers[i](lp_feat)
            if self.gate_encoder:
                x = self.apply_gate(x, lp_stage, i)
            x = stage(x)
            if i < len(self.stages) - 1:
                skip_connections.append(x)
                lp_skips.append(lp_stage)
                x = self.pooling_layers[i](x)
                lp_feat = torch.nn.functional.interpolate(lp_feat, size=x.shape[-2:], mode='bilinear', align_corners=False)
        
        if self.return_lp:
            return x, skip_connections, lp_skips
        else:
            return x, skip_connections


class XTinyMonoV2GatedEncDecUNetV1(XTinyMonoV2GatedEncUNetV1):
    def __init__(self, in_channels=1, num_classes=2, img_size=(256, 256), init_filters=1, max_filters=2, deep_supervision=True):
        super().__init__(in_channels, num_classes, img_size, init_filters, max_filters, deep_supervision)
        
        self.encoder.return_lp = True

        num_stages = 7 if max(img_size[0], img_size[1]) <= 256 else 8
        filters = [min(max_filters, init_filters * 2**i) for i in range(num_stages)]

        self.decoder = XTinyGatedDecoderV1(self.encoder, num_classes, filters, deep_supervision)

        self.initialize_weights()
    
    def forward(self, x):
        x, skip_connections, lp_skips = self.encoder(x)
        x = self.decoder(x, skip_connections, lp_skips)
        return x

class XTinyGatedDecoderV1(XTinyGatedDecoder):
    def __init__(self, encoder, num_classes=2, filters=[], deep_supervision=True):
        super().__init__(encoder, num_classes, filters, deep_supervision)
        
        stages = []
        num_stages = len(encoder.stages)
        for i in range(1, num_stages):
            stages.append(nn.Sequential(
                nn.Conv2d(filters[-(i+1)]*2, filters[-(i+1)], kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(filters[-(i+1)], eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
                nn.LeakyReLU(negative_slope=0.01, inplace=True)
            ))
        
        self.stages = nn.ModuleList(stages)
        self.lp_proj_layers = nn.ModuleList([nn.Conv2d(encoder.mono2d.out_channels, filters[-(i+1)], kernel_size=1, stride=1, bias=False) for i in range(1, num_stages)])
        self.gate_weights = nn.ParameterList([nn.Parameter(torch.ones(filters[-(i+1)])) for i in range(1, len(encoder.stages))])
        self.gate_biases = nn.ParameterList([nn.Parameter(torch.zeros(filters[-(i+1)])) for i in range(1, len(encoder.stages))])
        self.alphas = nn.ParameterList([nn.Parameter(torch.randn(filters[-(i+1)])) for i in range(1, len(encoder.stages))])
        
    def forward(self, x, skip_connections, lp_skips):
        outputs = []
        for i in range(len(self.stages)):
            x = self.transpose_convs[i](x)
            x = torch.cat([x, skip_connections[-(i+1)]], dim=1)
            x = self.stages[i](x)
            lp_stage = self.lp_proj_layers[i](lp_skips[-(i+1)])
            x = self.apply_gate(x, lp_stage, i)
            outputs.append(self.ds_seg_heads[i](x))
        
        if self.deep_supervision:
            return outputs
        else:
            return outputs[-1]
    
    def apply_gate(self, x: torch.Tensor, lp_features: torch.Tensor, layer_idx: int) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_weights[layer_idx].view(1, -1, 1, 1) * lp_features + self.gate_biases[layer_idx].view(1, -1, 1, 1))
        return x * (1 + self.alphas[layer_idx].view(1, -1, 1, 1) * gate)


class XTinyMonoV2GatedDecUNet(XTinyMonoV2GatedEncDecUNet):
    def __init__(self, in_channels=1, num_classes=2, img_size=(256, 256), init_filters=1, max_filters=2, deep_supervision=True):
        super().__init__(in_channels, num_classes, img_size, init_filters, max_filters, deep_supervision)
        
        self.encoder.gate_encoder = False


class XTinyMonoV2GatedDecUNetV1(XTinyMonoV2GatedEncDecUNetV1):
    def __init__(self, in_channels=1, num_classes=2, img_size=(256, 256), init_filters=1, max_filters=2, deep_supervision=True):
        super().__init__(in_channels, num_classes, img_size, init_filters, max_filters, deep_supervision)
        
        self.encoder.gate_encoder = False


if __name__ == "__main__":
    x = torch.randn((1,1,256,256))
    model = XTinyUNet(in_channels=1, num_classes=2, img_size=(256, 256), deep_supervision=True)
    print("Sample deep supervision outputs")
    for output in model(x):
        print(output.shape)
    
    # total params
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")