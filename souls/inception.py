"""
InceptionSoul - Inception-style multi-branch blocks with U-Net skips.

Combines multi-branch inception blocks (1x1, 3x3, 5x5, dilated 3x3) with
an encoder/decoder (U-Net) scaffold to create large, multi-scale patterns
on the n-dimensional unit sphere.
"""

import torch
import torch.nn.functional as F

from .base import Soul


class InceptionSoul(Soul):
    """
    Inception-style U-Net with sphere-respecting updates.
    
    Structure:
    - Encoder: inception blocks with downsampling to grow receptive field.
    - Bottleneck: extra inception blocks at the coarsest scale.
    - Decoder: upsample + inception blocks with skip connections.
    - Output is treated as tangent velocity and mapped back to the sphere.
    """
    
    def __init__(
        self,
        depth=3,
        base_channels=16,
        bottleneck_blocks=2,
        drift_magnitude=0.006,
        momentum=0.7,
        step_size=0.3,
        skip_strength=0.6,
        nonlinearity_strength=1.0,
        sphere_dim=4,
        use_expmap=True,
        device=None,
    ):
        self.depth = depth
        self.base_channels = base_channels
        self.bottleneck_blocks = bottleneck_blocks
        self.step_size = step_size
        self.skip_strength = skip_strength
        self.nonlinearity_strength = nonlinearity_strength
        self.use_expmap = use_expmap
        
        # Will be filled in during kernel init for decoder alignment
        self.encoder_blocks = []
        self.decoder_blocks = []
        self.decoder_levels = []
        self.bottleneck = []
        
        super().__init__(
            padding=2,  # safe pad for up to 5x5
            drift_magnitude=drift_magnitude,
            momentum=momentum,
            sphere_dim=sphere_dim,
            device=device,
        )
    
    def _inception_branch_channels(self, out_ch):
        """Split output channels across branches."""
        b1 = max(1, out_ch // 5)
        b3 = max(1, out_ch // 3)
        b5 = max(1, out_ch // 4)
        bd = max(1, out_ch - (b1 + b3 + b5))
        return b1, b3, b5, bd
    
    def _init_block(self, in_ch, out_ch, dilation):
        """Create inception block kernels and record layout."""
        b1, b3, b5, bd = self._inception_branch_channels(out_ch)
        
        # 1x1
        k1 = torch.randn(b1, in_ch, 1, 1, device=self.device)
        k1 = k1 - k1.mean(dim=(2, 3), keepdim=True)
        k1 = k1 / (k1.std() + 1e-6)
        
        # 3x3
        k3 = torch.randn(b3, in_ch, 3, 3, device=self.device)
        k3 = k3 - k3.mean(dim=(2, 3), keepdim=True)
        k3 = k3 / (k3.std() * 9 + 1e-6)
        
        # 5x5
        k5 = torch.randn(b5, in_ch, 5, 5, device=self.device)
        k5 = k5 - k5.mean(dim=(2, 3), keepdim=True)
        k5 = k5 / (k5.std() * 25 + 1e-6)
        
        # Dilated 3x3
        kd = torch.randn(bd, in_ch, 3, 3, device=self.device)
        kd = kd - kd.mean(dim=(2, 3), keepdim=True)
        kd = kd / (kd.std() * 9 + 1e-6)
        
        start_idx = len(self.kernels) if hasattr(self, "kernels") else 0
        
        kernels = [k1, k3, k5, kd]
        for k in kernels:
            self.kernels.append(k) if hasattr(self, "kernels") else None
        
        layout = {
            "k1": start_idx,
            "k3": start_idx + 1,
            "k5": start_idx + 2,
            "kd": start_idx + 3,
            "dilation": dilation,
            "out_channels": out_ch,
        }
        return kernels, layout
    
    def _initialize_kernels(self):
        """
        Build all inception block kernels and record layouts for encoder/decoder.
        """
        self.kernels = []
        self.encoder_blocks = []
        self.decoder_blocks = []
        self.decoder_levels = []
        self.bottleneck = []
        
        # Compute channel dims for encoder levels
        channel_dims = []
        for i in range(self.depth):
            in_dim = self.sphere_dim if i == 0 else channel_dims[-1][1]
            out_dim = self.base_channels * (2 ** i)
            channel_dims.append((in_dim, out_dim))
        
        # Encoder blocks
        for i, (in_dim, out_dim) in enumerate(channel_dims):
            _, layout = self._init_block(in_dim, out_dim, dilation=1 + i)
            self.encoder_blocks.append(layout)
        
        # Bottleneck blocks at deepest level
        bottleneck_dim = channel_dims[-1][1]
        for i in range(self.bottleneck_blocks):
            _, layout = self._init_block(bottleneck_dim, bottleneck_dim, dilation=2 + i)
            self.bottleneck.append(layout)
        
        # Decoder blocks (reverse levels)
        prev_out = bottleneck_dim
        for level in reversed(range(self.depth)):
            skip_ch = channel_dims[level][1]
            target_out = channel_dims[level][0] if level > 0 else self.sphere_dim
            in_ch = prev_out + skip_ch
            _, layout = self._init_block(in_ch, target_out, dilation=1 + level)
            self.decoder_blocks.append(layout)
            self.decoder_levels.append(level)
            prev_out = target_out
        
        return self.kernels
    
    def _apply_block(self, x, layout):
        """Run an inception block given layout indices."""
        b1 = F.conv2d(x, self.kernels[layout["k1"]])
        
        pad3 = 1
        x3 = F.pad(x, (pad3, pad3, pad3, pad3), mode="circular")
        b3 = F.conv2d(x3, self.kernels[layout["k3"]])
        
        pad5 = 2
        x5 = F.pad(x, (pad5, pad5, pad5, pad5), mode="circular")
        b5 = F.conv2d(x5, self.kernels[layout["k5"]])
        
        dil = layout["dilation"]
        padd = dil
        xd = F.pad(x, (padd, padd, padd, padd), mode="circular")
        bd = F.conv2d(xd, self.kernels[layout["kd"]], dilation=dil)
        
        y = torch.cat([b1, b3, b5, bd], dim=1)
        
        # Match expected out channels if off by 1 due to rounding
        if y.shape[1] > layout["out_channels"]:
            y = y[:, : layout["out_channels"], :, :]
        elif y.shape[1] < layout["out_channels"]:
            # Pad with zeros to hit target
            pad_ch = layout["out_channels"] - y.shape[1]
            y = torch.cat([y, torch.zeros(y.shape[0], pad_ch, y.shape[2], y.shape[3], device=y.device)], dim=1)
        
        y = torch.tanh(y * self.nonlinearity_strength) / max(self.nonlinearity_strength, 0.1)
        return y
    
    def apply(self, image, residual_alpha=0.2):
        x0 = image
        if x0.device != self.device:
            x0 = x0.to(self.device)
        
        x = x0.unsqueeze(0)  # (1, C, H, W)
        encoder_features = []
        spatial_sizes = []
        
        # Encoder
        for i, layout in enumerate(self.encoder_blocks):
            x = self._apply_block(x, layout)
            encoder_features.append(x)
            spatial_sizes.append(x.shape[-2:])
            if i < self.depth - 1:
                x = F.avg_pool2d(x, kernel_size=2, stride=2)
        
        # Bottleneck
        for layout in self.bottleneck:
            x = self._apply_block(x, layout)
        
        # Decoder
        for idx, layout in enumerate(self.decoder_blocks):
            level = self.decoder_levels[idx]
            # Upsample to match skip size (except for first if already aligned)
            target_size = encoder_features[level].shape[-2:]
            if x.shape[-2:] != target_size:
                x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
            
            skip = encoder_features[level] * self.skip_strength
            x = torch.cat([x, skip], dim=1)
            x = self._apply_block(x, layout)
        
        # Remove batch dim
        u = x.squeeze(0)
        
        # Tangent projection and sphere move
        v = self.tangent_project(x0, u)
        effective_step = self.step_size * residual_alpha
        
        if self.use_expmap:
            x_new = self.sphere_expmap(x0, v, step=effective_step)
        else:
            x_new = self.sphere_retract(x0, v, step=effective_step)
        
        x_new = self.sphere_normalize(x_new, dim=0)
        return x_new
    
    def get_soul_sliders(self):
        return [
            {"label": "Step Size", "value_attr": "step_size", "min_value": 0.05, "max_value": 1.0},
            {"label": "Skip Strength", "value_attr": "skip_strength", "min_value": 0.0, "max_value": 1.0},
            {"label": "Nonlinearity", "value_attr": "nonlinearity_strength", "min_value": 0.2, "max_value": 3.0},
        ]
