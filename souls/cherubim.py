"""
Cherubim - Encoder-decoder convolution soul with multi-scale processing.

Named after the second-highest order of angels, this soul uses a classic
encoder-decoder architecture to process information at multiple scales,
encouraging different behaviors in different spatial regions.
"""

import torch
import torch.nn.functional as F

from .base import Soul


class Cherubim(Soul):
    """
    Encoder-decoder architecture with multi-scale processing.
    
    Architecture:
    - Encode: 3x3 conv (expand channels) + 2x2 avg pool (downsample)
    - Repeat for 'depth' levels
    - Decode: 2x2 upsample + 3x3 conv (reduce channels)
    - Repeat back to original size
    
    This creates a bottleneck that encourages different spatial behaviors
    and multi-scale feature processing.
    """
    
    def __init__(self, depth=3, latent_dim=16, drift_magnitude=0.002, 
                 momentum=0.7, step_size=0.2, sphere_dim=3, use_expmap=True, 
                 skip_strength=0.3, nonlinearity_strength=1.0, 
                 bottleneck_iterations=2, device=None):
        """
        Initialize encoder-decoder convolution processor.
        
        Args:
            depth: Number of encoder/decoder levels (default: 3)
            latent_dim: Channel dimension at deepest level (default: 16)
            drift_magnitude: Magnitude of drift direction vector
            momentum: Momentum factor for drift direction updates (0-1)
            step_size: Step size for sphere updates (default: 0.2)
            sphere_dim: Dimensionality of the unit sphere (default: 3)
            use_expmap: Use exponential map (True) or simple retraction (False)
            skip_strength: Strength of skip connections (0=none, 1=full) (default: 0.3)
            nonlinearity_strength: Strength of tanh nonlinearity (default: 1.0)
            bottleneck_iterations: Extra processing iterations at bottleneck (default: 2)
            device: PyTorch device (cuda, mps, or cpu)
        """
        self.depth = depth
        self.latent_dim = latent_dim
        self.step_size = step_size
        self.use_expmap = use_expmap
        self.skip_strength = skip_strength
        self.nonlinearity_strength = nonlinearity_strength
        self.bottleneck_iterations = bottleneck_iterations
        
        # Calculate channel dimensions at each level
        # Smoothly interpolate from sphere_dim to latent_dim
        self.channel_dims = self._compute_channel_dims(sphere_dim, latent_dim, depth)
        
        super().__init__(padding=1, drift_magnitude=drift_magnitude,
                        momentum=momentum, sphere_dim=sphere_dim, device=device)
    
    def _compute_channel_dims(self, input_dim, latent_dim, depth):
        """
        Compute channel dimensions at each level.
        
        Linearly interpolate from input_dim to latent_dim over depth levels.
        
        Returns:
            List of (in_channels, out_channels) for each encoder level
        """
        dims = []
        for i in range(depth):
            # Linear interpolation
            t = (i + 1) / depth
            out_dim = int(input_dim + t * (latent_dim - input_dim))
            in_dim = input_dim if i == 0 else dims[-1][1]
            dims.append((in_dim, out_dim))
        return dims
    
    def _initialize_kernels(self):
        """
        Initialize encoder, decoder, and bottleneck kernels.
        
        Returns list of:
        - encoder_kernels: depth levels of (in_ch, out_ch, 3, 3)
        - bottleneck_kernels: bottleneck_iterations of (latent_dim, latent_dim, 3, 3)
        - decoder_kernels: depth levels of (combined_ch, in_ch, 3, 3) (with skip connections)
        """
        kernels = []
        
        # Encoder kernels (expand channels)
        for in_dim, out_dim in self.channel_dims:
            kernel = torch.randn(out_dim, in_dim, 3, 3, device=self.device)
            
            # Zero-mean for each output channel
            kernel = kernel - kernel.mean(dim=(-1, -2), keepdim=True)
            
            # Normalize
            kernel = kernel / (kernel.std() * 9 + 1e-6)
            
            kernels.append(kernel)
        
        # Bottleneck processing kernels (same dimension, deeper processing)
        bottleneck_dim = self.channel_dims[-1][1]  # Output of last encoder
        for _ in range(self.bottleneck_iterations):
            kernel = torch.randn(bottleneck_dim, bottleneck_dim, 3, 3, device=self.device)
            kernel = kernel - kernel.mean(dim=(-1, -2), keepdim=True)
            kernel = kernel / (kernel.std() * 9 + 1e-6)
            kernels.append(kernel)
        
        # Decoder kernels (reduce channels, with skip connections)
        # Skip connections concatenate encoder features with decoder features
        decoder_dims = list(reversed(self.channel_dims))
        
        for i, (encoder_in, encoder_out) in enumerate(decoder_dims):
            # Decoder goes: latent_dim -> ... -> sphere_dim
            # decoder_dims[0] = last encoder layer (encoder_out = latent_dim)
            # decoder_dims[-1] = first encoder layer (encoder_in = sphere_dim)
            
            # For first decoder layer: input is just bottleneck output
            if i == 0:
                kernel_in_dim = encoder_out  # bottleneck dimension
                kernel_out_dim = encoder_in  # reduce back
            else:
                # For subsequent layers:
                # Input = previous decoder output + skip connection from encoder
                prev_decoder_out = decoder_dims[i-1][0]  # Previous layer's output channels
                
                # Skip connection comes from matching encoder level
                # At decoder i, we skip from encoder level (depth - i - 1)
                skip_encoder_idx = self.depth - i - 1
                skip_channels = self.channel_dims[skip_encoder_idx][1]  # Output channels from that encoder level
                
                kernel_in_dim = prev_decoder_out + skip_channels
                kernel_out_dim = encoder_in
            
            kernel = torch.randn(kernel_out_dim, kernel_in_dim, 3, 3, device=self.device)
            
            # Zero-mean
            kernel = kernel - kernel.mean(dim=(-1, -2), keepdim=True)
            
            # Normalize
            kernel = kernel / (kernel.std() * 9 + 1e-6)
            
            kernels.append(kernel)
        
        return kernels
    
    def _pool(self, x):
        """
        Average pooling 2x2 with stride 2.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Pooled tensor (B, C, H/2, W/2)
        """
        return F.avg_pool2d(x, kernel_size=2, stride=2)
    
    def _upsample(self, x):
        """
        Upsample 2x using bilinear interpolation.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Upsampled tensor (B, C, H*2, W*2)
        """
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
    
    def apply(self, image, residual_alpha=0.2):
        """
        Apply encoder-decoder processing with sphere constraint.
        
        Strategy:
        1. Encode: conv + pool (downsampling + channel expansion)
        2. Bottleneck: extra processing iterations at lowest resolution
        3. Decode: upsample + conv with skip connections (U-Net style)
        4. Apply sphere constraint via tangent space update
        
        Args:
            image: Input image tensor (C, H, W) where C=sphere_dim
            residual_alpha: Controls step size blending
            
        Returns:
            Processed image tensor (C, H, W) on unit sphere
        """
        # Ensure on device and sphere
        x = image
        if x.device != self.device:
            x = x.to(self.device)
        
        x_orig = x  # Save for tangent projection at the end
        
        # Add batch dimension
        x = x.unsqueeze(0)  # (1, C, H, W)
        
        # Keep track of spatial dimensions and encoder features for skip connections
        spatial_sizes = []
        encoder_features = []
        
        # ===== ENCODER PATH =====
        for i in range(self.depth):
            kernel = self.kernels[i]
            
            # Apply convolution with circular padding
            x_padded = F.pad(x, (1, 1, 1, 1), mode='circular')
            x = F.conv2d(x_padded, kernel, padding=0)
            
            # Apply nonlinearity with adjustable strength
            # For strength < 1: less saturation, more linear
            # For strength > 1: more saturation, more nonlinear
            x = torch.tanh(x * self.nonlinearity_strength) / max(self.nonlinearity_strength, 0.1)
            
            # Save for skip connection (before pooling)
            encoder_features.append(x)
            spatial_sizes.append(x.shape[2:])
            
            # Downsample (except on last layer - stay at bottleneck)
            if i < self.depth - 1:
                x = self._pool(x)
        
        # ===== BOTTLENECK PROCESSING =====
        # Extra iterations at lowest resolution for richer processing
        bottleneck_start_idx = self.depth
        for i in range(self.bottleneck_iterations):
            kernel_idx = bottleneck_start_idx + i
            kernel = self.kernels[kernel_idx]
            
            x_padded = F.pad(x, (1, 1, 1, 1), mode='circular')
            x = F.conv2d(x_padded, kernel, padding=0)
            x = torch.tanh(x * self.nonlinearity_strength) / max(self.nonlinearity_strength, 0.1)
        
        # ===== DECODER PATH =====
        decoder_start_idx = self.depth + self.bottleneck_iterations
        
        for i in range(self.depth):
            kernel_idx = decoder_start_idx + i
            kernel = self.kernels[kernel_idx]
            
            # Upsample (except on first decoder layer)
            if i > 0:
                # Upsample to match encoder level size
                target_size = spatial_sizes[self.depth - i - 1]
                x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
            
            # Apply skip connection (U-Net style)
            # Concatenate encoder features from corresponding level
            # Skip connection mapping:
            # decoder i=0 -> no skip (we're at bottleneck level)
            # decoder i=1 -> encoder level depth-2 (second-to-last encoder)
            # decoder i=2 -> encoder level depth-3, etc.
            if i > 0:
                # Get skip features from corresponding encoder level
                # Encoder features are stored in order [0, 1, 2, ...]
                # We want to match them in reverse during decoding
                encoder_idx = self.depth - i - 1
                skip_features = encoder_features[encoder_idx]
                
                # Scale skip connection with adjustable strength
                skip_features = skip_features * self.skip_strength
                x = torch.cat([x, skip_features], dim=1)
            
            # Apply convolution with circular padding
            x_padded = F.pad(x, (1, 1, 1, 1), mode='circular')
            x = F.conv2d(x_padded, kernel, padding=0)
            
            # Apply nonlinearity
            x = torch.tanh(x * self.nonlinearity_strength) / max(self.nonlinearity_strength, 0.1)
        
        # Remove batch dimension
        x = x.squeeze(0)  # (C, H, W)
        
        # ===== SPHERE CONSTRAINT =====
        # Treat output as velocity field in tangent space
        u = x
        
        # Project to tangent space at original position
        v = self.tangent_project(x_orig, u)
        
        # Move on sphere using exponential map or retraction
        effective_step = self.step_size * residual_alpha
        
        if self.use_expmap:
            x_new = self.sphere_expmap(x_orig, v, step=effective_step)
        else:
            x_new = self.sphere_retract(x_orig, v, step=effective_step)
        
        # Final normalization
        x_new = self.sphere_normalize(x_new, dim=0)
        
        return x_new
    
    def get_soul_sliders(self):
        """Return Cherubim-specific sliders."""
        return [
            {
                "label": "Step Size",
                "value_attr": "step_size",
                "min_value": 0.01,
                "max_value": 1.0
            },
            {
                "label": "Skip Strength",
                "value_attr": "skip_strength",
                "min_value": 0.0,
                "max_value": 1.0
            },
            {
                "label": "Nonlinearity",
                "value_attr": "nonlinearity_strength",
                "min_value": 0.1,
                "max_value": 3.0
            }
        ]

