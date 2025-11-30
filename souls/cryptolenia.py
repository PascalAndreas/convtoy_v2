"""
CryptoLenia - Odd, anisotropic Lenia-inspired waves on the n-sphere.

Adds rotating anisotropic kernels, band-pass growth, phase noise, and jittered
rolls to keep patterns weird and restless.
"""

import torch
import torch.nn.functional as F

from .base import Soul


def _aniso_kernel(size, angle, stretch, device):
    """Anisotropic Gaussian rotated by angle."""
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, size, device=device),
        torch.linspace(-1, 1, size, device=device),
        indexing="ij",
    )
    ca, sa = torch.cos(angle), torch.sin(angle)
    xr = ca * x + sa * y
    yr = -sa * x + ca * y
    k = torch.exp(-((xr ** 2) / (2 * (0.2 ** 2)) + (yr ** 2) / (2 * (0.2 * stretch) ** 2)))
    k = k / (k.sum() + 1e-6)
    return k


class CryptoLenia(Soul):
    """
    Anisotropic Lenia: rotating kernels, band-pass growth, and phase noise.
    """
    
    def __init__(
        self,
        sizes=(5, 9, 13),
        stretch=2.2,
        angle_speed=0.35,
        band_center=0.4,
        band_width=0.18,
        growth_gain=1.3,
        feedback=0.4,
        jitter=0.12,
        roll_max=3,
        drift_magnitude=0.004,
        momentum=0.6,
        step_size=0.32,
        sphere_dim=4,
        use_expmap=True,
        device=None,
    ):
        self.sizes = list(sizes)
        self.stretch = stretch
        self.angle_speed = angle_speed
        self.band_center = band_center
        self.band_width = band_width
        self.growth_gain = growth_gain
        self.feedback = feedback
        self.jitter = jitter
        self.roll_max = roll_max
        self.step_size = step_size
        self.use_expmap = use_expmap
        
        self.angle = 0.0
        self.prev_velocity = None
        
        super().__init__(
            padding=max(self.sizes),
            drift_magnitude=drift_magnitude,
            momentum=momentum,
            sphere_dim=sphere_dim,
            device=device,
        )
    
    def _initialize_kernels(self):
        """Create rotated anisotropic kernels per size and channel (diagonal)."""
        kernels = []
        for size in self.sizes:
            k = _aniso_kernel(size, torch.tensor(self.angle, device=self.device), self.stretch, self.device)
            kernel = torch.zeros(self.sphere_dim, self.sphere_dim, size, size, device=self.device)
            for c in range(self.sphere_dim):
                kernel[c, c] = k
            kernels.append(kernel)
        return kernels
    
    def _band_growth(self, mag):
        """Band-pass growth curve."""
        return torch.exp(-((mag - self.band_center) ** 2) / (2 * self.band_width ** 2)) - 0.5
    
    def _refresh_kernels(self):
        """Rotate kernels a bit each step to keep anisotropy moving."""
        self.angle = (self.angle + self.angle_speed * 0.05) % (2 * torch.pi)
        self.kernels = []
        for size in self.sizes:
            k = _aniso_kernel(size, torch.tensor(self.angle, device=self.device), self.stretch, self.device)
            kernel = torch.zeros(self.sphere_dim, self.sphere_dim, size, size, device=self.device)
            for c in range(self.sphere_dim):
                kernel[c, c] = k
            self.kernels.append(kernel)
    
    def apply(self, image, residual_alpha=0.2):
        x0 = image
        if x0.device != self.device:
            x0 = x0.to(self.device)
        
        if self.prev_velocity is None or self.prev_velocity.shape != x0.shape:
            self.prev_velocity = torch.zeros_like(x0, device=self.device)
        
        # Rotate kernels slowly
        self._refresh_kernels()
        
        x = x0.unsqueeze(0)
        total_v = torch.zeros_like(x0)
        total_g = torch.zeros_like(x0[0])
        
        for kernel in self.kernels:
            pad = kernel.shape[-1] // 2
            xp = F.pad(x, (pad, pad, pad, pad), mode="circular")
            conv = F.conv2d(xp, kernel, padding=0).squeeze(0)
            mag = torch.linalg.vector_norm(conv, dim=0)
            growth = self._band_growth(mag)
            v = self.tangent_project(x0, conv) * growth.unsqueeze(0)
            total_v = total_v + v
            total_g = total_g + growth
        
        total_v = total_v / max(len(self.kernels), 1)
        total_v = total_v + self.prev_velocity * self.feedback
        
        if self.jitter > 0:
            noise = torch.randn_like(total_v) * self.jitter
            noise = self.tangent_project(x0, noise)
            total_v = total_v + noise
        
        if self.roll_max > 0:
            sx = int(torch.randint(-self.roll_max, self.roll_max + 1, (1,)).item())
            sy = int(torch.randint(-self.roll_max, self.roll_max + 1, (1,)).item())
            total_v = torch.roll(total_v, shifts=(sy, sx), dims=(1, 2))
        
        total_v = total_v * (self.growth_gain * residual_alpha)
        
        if self.use_expmap:
            x_new = self.sphere_expmap(x0, total_v, step=self.step_size)
        else:
            x_new = self.sphere_retract(x0, total_v, step=self.step_size)
        
        x_new = self.sphere_normalize(x_new, dim=0)
        self.prev_velocity = total_v.detach()
        return x_new
    
    def get_soul_sliders(self):
        return [
            {"label": "Step Size", "value_attr": "step_size", "min_value": 0.05, "max_value": 1.0},
            {"label": "Band Center", "value_attr": "band_center", "min_value": 0.0, "max_value": 1.0},
            {"label": "Band Width", "value_attr": "band_width", "min_value": 0.01, "max_value": 0.6},
            {"label": "Growth Gain", "value_attr": "growth_gain", "min_value": 0.1, "max_value": 3.0},
            {"label": "Feedback", "value_attr": "feedback", "min_value": 0.0, "max_value": 0.9},
            {"label": "Jitter", "value_attr": "jitter", "min_value": 0.0, "max_value": 0.4},
            {"label": "Roll Max", "value_attr": "roll_max", "min_value": 0, "max_value": 8},
        ]
