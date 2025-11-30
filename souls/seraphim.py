"""
Seraphim - Multi-layer convolution soul implementation.

Named after the highest order of angels, this soul uses multiple layers
of convolution to create complex, heavenly transformations.
"""

import torch
import torch.nn.functional as F

from .base import Soul


class Seraphim(Soul):
    """
    Multi-layer convolution processor with residual connections.
    
    Applies multiple sequential convolution layers with tanh nonlinearity
    and residual blending to create complex spatial transformations.
    """
    
    def __init__(self, kernel_size=7, num_layers=5, drift_magnitude=0.002, 
                 momentum=0.7, step_size=0.2, sphere_dim=3, use_expmap=True, device=None):
        """
        Initialize multi-layer convolution processor with sphere dynamics.
        
        Args:
            kernel_size: Size of convolution kernels (default: 7)
            num_layers: Number of sequential convolution layers (default: 5)
            drift_magnitude: Magnitude of drift direction vector
            momentum: Momentum factor for drift direction updates (0-1)
            step_size: Step size for sphere updates (default: 0.2)
            sphere_dim: Dimensionality of the unit sphere (default: 3)
            use_expmap: Use exponential map (True) or simple retraction (False)
            device: PyTorch device (cuda, mps, or cpu)
        """
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.step_size = step_size
        self.use_expmap = use_expmap
        super().__init__(padding=kernel_size // 2, drift_magnitude=drift_magnitude,
                        momentum=momentum, sphere_dim=sphere_dim, device=device)
    
    def _initialize_kernels(self):
        """Generate random convolution kernels for each layer (sphere_dim channels)."""
        kernels = []
        for _ in range(self.num_layers):
            # Kernels now operate on sphere_dim channels instead of RGB (3)
            # Create directly on device for efficiency
            kernel = torch.randn(self.sphere_dim, self.sphere_dim, 
                               self.kernel_size, self.kernel_size,
                               device=self.device)
            
            # CRITICAL: Kill DC mode - make each kernel zero-mean
            # This prevents "single color" collapse
            kernel = kernel - kernel.mean(dim=(-1, -2), keepdim=True)
            
            # Normalize to control spectral radius (prevent explosion)
            kernel = kernel / (kernel.std() * (self.kernel_size ** 2) + 1e-6)
            
            kernels.append(kernel)
        return kernels
    
    def apply(self, image, residual_alpha=0.2):
        """
        Apply sphere-constrained convolution using tangent-space updates.
        
        This implements the "tangent update + retraction/expmap" approach:
        1. Compute convolution (velocity field in ambient space)
        2. Project to tangent space at current point
        3. Move along sphere using retraction or exponential map
        
        Args:
            image: Input image tensor (C, H, W) where C=sphere_dim, on any device
                   Each pixel is a unit vector: ||image[:, h, w]|| = 1
            residual_alpha: Controls step size blending (0=no movement, 1=full step)
            
        Returns:
            Processed image tensor (C, H, W) on device, still on unit sphere
        """
        # Image should already be on device, but ensure it
        x = image
        if x.device != self.device:
            x = x.to(self.device)
        
        # Only normalize once at the start (input should already be normalized)
        # Skip if already normalized to save computation
        
        pad_size = self.padding
        effective_step = self.step_size * residual_alpha
        
        # Add batch dimension once for all layers
        x = x.unsqueeze(0)
        
        # Apply each layer sequentially with sphere constraint
        for kernel in self.kernels:
            # Store input for tangent projection
            x_in = x.squeeze(0)
            
            # Apply circular padding so edges wrap around
            x_padded = F.pad(x, (pad_size, pad_size, pad_size, pad_size), 
                           mode='circular')
            
            # Apply convolution to get ambient velocity field
            u = F.conv2d(x_padded, kernel, padding=0).squeeze(0)
            
            # Project onto tangent space at x_in
            # This ensures the velocity is orthogonal to the current position
            v = self.tangent_project(x_in, u)
            
            # Move on the sphere using exponential map or retraction
            if self.use_expmap:
                x_new = self.sphere_expmap(x_in, v, step=effective_step)
            else:
                x_new = self.sphere_retract(x_in, v, step=effective_step)
            
            # Update for next layer (add batch dim back)
            x = x_new.unsqueeze(0)
        
        # Remove batch dimension
        x = x.squeeze(0)
        
        # Final normalization to ensure we're exactly on sphere
        # This is important to prevent drift over many iterations
        x = self.sphere_normalize(x, dim=0)
        
        # Keep on device for efficiency
        return x
    
    def get_soul_sliders(self):
        """Return Seraphim-specific sliders"""
        return [
            {
                "label": "Step Size",
                "value_attr": "step_size",
                "min_value": 0.01,
                "max_value": 1.0
            }
        ]

