"""
Lenia-inspired smooth-life soul on the n-sphere.

This soul borrows from Lenia's continuous cellular automata:
- Multi-scale radial kernels generate smooth growth fields
- Growth drives tangent-space velocity, keeping pixels on the unit sphere
- Feedback memory + jittered rolls encourage waves and spirals
"""

import torch
import torch.nn.functional as F

from .base import Soul


def _radial_kernel(radius, device):
    """Build a normalized radial kernel with gentle ring emphasis."""
    size = radius * 2 + 1
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, size, device=device),
        torch.linspace(-1, 1, size, device=device),
        indexing="ij",
    )
    r = torch.sqrt(x * x + y * y)
    # Slight ring bump to avoid boring blobs
    ring = torch.exp(-((r - 0.5) ** 2) / (2 * 0.18 ** 2))
    core = torch.exp(-(r ** 2) / (2 * 0.35 ** 2))
    k = core * 0.6 + ring * 0.4
    k = k / (k.sum() + 1e-6)
    return k


class Lenia(Soul):
    """
    Smooth-life dynamics projected onto the sphere.
    
    Pipeline:
    1) Multi-scale radial convolutions build growth maps.
    2) Growth steers tangent-space velocity from convolved features.
    3) Feedback memory and random rolls inject waves and rotation.
    4) Exponential map / retraction keeps points on S^{dim-1}.
    """
    
    def __init__(
        self,
        radii=(3, 7, 11),
        growth_peak=0.35,
        growth_width=0.12,
        growth_strength=1.4,
        feedback=0.35,
        jitter_strength=0.08,
        warp_roll=2,
        drift_magnitude=0.004,
        momentum=0.65,
        step_size=0.35,
        sphere_dim=4,
        use_expmap=True,
        device=None,
    ):
        self.radii = list(radii)
        self.growth_peak = growth_peak
        self.growth_width = growth_width
        self.growth_strength = growth_strength
        self.feedback = feedback
        self.jitter_strength = jitter_strength
        self.warp_roll = warp_roll
        self.step_size = step_size
        self.use_expmap = use_expmap
        
        # Memory of previous velocity for wave propagation
        self.prev_velocity = None
        
        super().__init__(
            padding=max(self.radii),
            drift_magnitude=drift_magnitude,
            momentum=momentum,
            sphere_dim=sphere_dim,
            device=device,
        )
    
    def _initialize_kernels(self):
        """
        Build one kernel per scale, shared across channels (diagonal weight).
        """
        kernels = []
        for r in self.radii:
            base = _radial_kernel(r, self.device)
            size = base.shape[0]
            
            # Shape: (C_out, C_in, H, W) with diagonal emphasis
            kernel = torch.zeros(
                self.sphere_dim, self.sphere_dim, size, size, device=self.device
            )
            for c in range(self.sphere_dim):
                kernel[c, c] = base
            
            kernels.append(kernel)
        return kernels
    
    def _growth_response(self, conv_mag):
        """
        Lenia growth curve: bell around growth_peak with width growth_width.
        """
        return torch.exp(-((conv_mag - self.growth_peak) ** 2) / (2 * self.growth_width ** 2)) - 0.5
    
    def apply(self, image, residual_alpha=0.2):
        x0 = image
        if x0.device != self.device:
            x0 = x0.to(self.device)
        
        # Lazy init memory
        if self.prev_velocity is None or self.prev_velocity.shape != x0.shape:
            self.prev_velocity = torch.zeros_like(x0, device=self.device)
        
        x = x0.unsqueeze(0)  # (1, C, H, W)
        
        total_velocity = torch.zeros_like(x0)
        total_growth = torch.zeros_like(x0[0])  # scalar map
        
        for kernel in self.kernels:
            pad = kernel.shape[-1] // 2
            x_padded = F.pad(x, (pad, pad, pad, pad), mode="circular")
            conv = F.conv2d(x_padded, kernel, padding=0).squeeze(0)  # (C, H, W)
            
            conv_mag = torch.linalg.vector_norm(conv, dim=0)
            growth = self._growth_response(conv_mag)
            
            # Tangent-directed velocity from conv features
            v = self.tangent_project(x0, conv) * growth.unsqueeze(0)
            total_velocity = total_velocity + v
            total_growth = total_growth + growth
        
        # Normalize contributions
        num_k = len(self.kernels)
        total_velocity = total_velocity / max(num_k, 1)
        total_growth = total_growth / max(num_k, 1)
        
        # Add feedback memory for wave propagation
        total_velocity = total_velocity + self.prev_velocity * self.feedback
        
        # Jitter + roll warp to break symmetry
        if self.jitter_strength > 0:
            noise = torch.randn_like(total_velocity) * self.jitter_strength
            noise = self.tangent_project(x0, noise)
            total_velocity = total_velocity + noise
        
        if self.warp_roll > 0:
            self.warp_roll = int(self.warp_roll)
            shift_y = int(torch.randint(-self.warp_roll, self.warp_roll + 1, (1,)).item())
            shift_x = int(torch.randint(-self.warp_roll, self.warp_roll + 1, (1,)).item())
            total_velocity = torch.roll(total_velocity, shifts=(shift_y, shift_x), dims=(1, 2))
        
        # Scale by growth strength
        total_velocity = total_velocity * (self.growth_strength * residual_alpha)
        
        # Move on sphere
        if self.use_expmap:
            x_new = self.sphere_expmap(x0, total_velocity, step=self.step_size)
        else:
            x_new = self.sphere_retract(x0, total_velocity, step=self.step_size)
        
        x_new = self.sphere_normalize(x_new, dim=0)
        
        # Update memory (detach to avoid graph)
        self.prev_velocity = total_velocity.detach()
        
        return x_new
    
    def get_soul_sliders(self):
        return [
            {"label": "Step Size", "value_attr": "step_size", "min_value": 0.05, "max_value": 1.0},
            {"label": "Growth Peak", "value_attr": "growth_peak", "min_value": 0.0, "max_value": 1.0},
            {"label": "Growth Width", "value_attr": "growth_width", "min_value": 0.01, "max_value": 0.6},
            {"label": "Growth Strength", "value_attr": "growth_strength", "min_value": 0.1, "max_value": 3.0},
            {"label": "Feedback", "value_attr": "feedback", "min_value": 0.0, "max_value": 0.9},
            {"label": "Jitter", "value_attr": "jitter_strength", "min_value": 0.0, "max_value": 0.4},
            {"label": "Warp Roll", "value_attr": "warp_roll", "min_value": 0, "max_value": 8},
        ]
