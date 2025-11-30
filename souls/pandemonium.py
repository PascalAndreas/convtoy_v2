"""
Pandemonium - Chaotic, pruning, warp-heavy sphere soul.

This soul embraces unstable dynamics:
- Expands into a latent chaos field with logistic-map-driven channel gates
- Dynamically prunes/keeps channels each step (soft top-k gating)
- Multi-dilation convolutions for large-scale motion
- Flow-based warping + glitchy rolls for discontinuities
"""

import torch
import torch.nn.functional as F

from .base import Soul


class Pandemonium(Soul):
    """
    Chaotic latent-field processor with dynamic channel selection and warping.
    
    Pipeline:
    1) Expand sphere channels into latent_dim channels.
    2) Chaotic logistic map generates gates; soft top-k pruning keeps a subset.
    3) Multi-dilation convolutions (widen receptive field) with gated channels.
    4) Warp the latent grid using noise-driven flow + glitchy toroidal rolls.
    5) Collapse back to sphere space and move along the sphere manifold.
    """
    
    def __init__(
        self,
        latent_dim=32,
        num_blocks=4,
        drift_magnitude=0.01,
        momentum=0.65,
        step_size=0.35,
        selection_ratio=0.4,
        chaos_intensity=3.7,
        gate_sharpness=12.0,
        glitch_strength=0.25,
        warp_strength=0.18,
        pulse_mix=0.5,
        sphere_dim=4,
        use_expmap=True,
        device=None,
    ):
        """
        Initialize chaotic soul.
        
        Args:
            latent_dim: Width of latent chaos field.
            num_blocks: Number of gated conv blocks (each with its own dilation).
            selection_ratio: Fraction of channels to keep (softly) per step.
            chaos_intensity: Logistic map multiplier (3.5-3.99 yields chaos).
            gate_sharpness: Sigmoid steepness for pruning.
            glitch_strength: Strength of glitchy roll mixing.
            warp_strength: Magnitude of flow-based warp.
            pulse_mix: Blend between low-frequency pulse and raw latent.
            sphere_dim: Dimensionality of the sphere input/output.
        """
        self.latent_dim = latent_dim
        self.num_blocks = num_blocks
        self.step_size = step_size
        self.selection_ratio = selection_ratio
        self.chaos_intensity = chaos_intensity
        self.gate_sharpness = gate_sharpness
        self.glitch_strength = glitch_strength
        self.warp_strength = warp_strength
        self.pulse_mix = pulse_mix
        self.use_expmap = use_expmap
        
        # Precompute dilations for broader receptive fields
        self.dilations = [1 + i for i in range(num_blocks)]
        
        # Chaotic state (per-channel) drives gating; initialized in (0,1)
        self.chaos_state = None
        
        super().__init__(
            padding=1,
            drift_magnitude=drift_magnitude,
            momentum=momentum,
            sphere_dim=sphere_dim,
            device=device,
        )
    
    def _initialize_kernels(self):
        """
        Kernels:
        - 0: expand sphere_dim -> latent_dim (3x3)
        - 1..num_blocks: latent->latent (3x3) with varying dilation
        - last: collapse latent -> sphere_dim (1x1)
        """
        kernels = []
        
        # Expand
        expand = torch.randn(self.latent_dim, self.sphere_dim, 3, 3, device=self.device)
        expand = expand - expand.mean(dim=(-1, -2), keepdim=True)
        expand = expand / (expand.std() * 9 + 1e-6)
        kernels.append(expand)
        
        # Gated blocks
        for _ in range(self.num_blocks):
            k = torch.randn(self.latent_dim, self.latent_dim, 3, 3, device=self.device)
            k = k - k.mean(dim=(-1, -2), keepdim=True)
            k = k / (k.std() * 9 + 1e-6)
            kernels.append(k)
        
        # Collapse
        collapse = torch.randn(self.sphere_dim, self.latent_dim, 1, 1, device=self.device)
        collapse = collapse - collapse.mean(dim=(1,), keepdim=True)
        collapse = collapse / (collapse.std() + 1e-6)
        kernels.append(collapse)
        
        # Initialize chaotic gate state
        self.chaos_state = torch.rand(self.latent_dim, device=self.device) * 0.6 + 0.2
        
        return kernels
    
    def _chaotic_gates(self):
        """Update logistic map and derive soft top-k gates."""
        noise = torch.randn_like(self.chaos_state) * 0.02 * self.glitch_strength
        self.chaos_state = self.chaos_intensity * self.chaos_state * (1.0 - self.chaos_state) + noise
        self.chaos_state = torch.clamp(self.chaos_state, 1e-4, 0.999)
        
        # Soft top-k: threshold at selection_ratio percentile
        k = max(1, int(self.selection_ratio * self.latent_dim))
        values, _ = torch.topk(self.chaos_state, k)
        threshold = values.min()
        
        gates = torch.sigmoid((self.chaos_state - threshold) * self.gate_sharpness)
        return gates
    
    def _warp_latent(self, x):
        """Apply noise-driven flow warp to latent grid."""
        b, c, h, w = x.shape
        # Coarse noise flow
        flow_scale = max(2, min(h, w) // 8)
        flow = torch.randn(b, 2, h // flow_scale + 1, w // flow_scale + 1, device=x.device)
        flow = F.interpolate(flow, size=(h, w), mode="bilinear", align_corners=False)
        flow = torch.tanh(flow) * self.warp_strength
        
        # Base grid in [-1,1]
        y, x_coords = torch.meshgrid(
            torch.linspace(-1, 1, h, device=x.device),
            torch.linspace(-1, 1, w, device=x.device),
            indexing="ij",
        )
        grid = torch.stack((x_coords, y), dim=-1).unsqueeze(0).expand(b, -1, -1, -1)
        
        # Add flow to grid (swap order: dx, dy)
        grid = grid + flow.permute(0, 2, 3, 1)
        # Wrap grid into [-1, 1] to emulate toroidal sampling
        grid = (grid + 1.0) % 2.0 - 1.0
        
        warped = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        return warped
    
    def _glitch_roll(self, x):
        """Toroidal roll jitter for glitchy discontinuities."""
        shift_x = int((self.chaos_state.mean() * 17 * self.glitch_strength)) % x.shape[-1]
        shift_y = int((self.chaos_state.std() * 19 * self.glitch_strength)) % x.shape[-2]
        rolled = torch.roll(x, shifts=(shift_y, shift_x), dims=(2, 3))
        return rolled
    
    def apply(self, image, residual_alpha=0.2):
        """
        Apply chaotic processing on the sphere.
        """
        x0 = image
        if x0.device != self.device:
            x0 = x0.to(self.device)
        
        # Ensure chaos state on device
        if self.chaos_state is None or self.chaos_state.device != self.device:
            self.chaos_state = torch.rand(self.latent_dim, device=self.device) * 0.6 + 0.2
        
        gates = self._chaotic_gates()
        
        # Expand
        x = x0.unsqueeze(0)
        expand_kernel = self.kernels[0]
        x = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="circular"), expand_kernel)
        
        # Apply gates (channel-wise) softly
        x = x * gates.view(1, -1, 1, 1)
        
        # Multi-dilation gated blocks
        for i in range(self.num_blocks):
            k_idx = 1 + i
            kernel = self.kernels[k_idx]
            
            dilation = self.dilations[i]
            pad = dilation
            x_padded = F.pad(x, (pad, pad, pad, pad), mode="circular")
            
            # Velocity in latent space
            v = F.conv2d(x_padded, kernel, dilation=dilation)
            
            # Nonlinearity with pulse mix for low/high frequency blend
            v_nl = torch.tanh(v * 0.8)
            if self.pulse_mix > 0:
                # Low-frequency pulse from pooled latent
                pooled = F.avg_pool2d(x, kernel_size=4, stride=2, padding=1)
                pooled = F.interpolate(pooled, size=x.shape[-2:], mode="bilinear", align_corners=False)
                v_nl = v_nl * (1 - self.pulse_mix) + pooled * self.pulse_mix
            
            # Apply gating again to encourage sparsity drift
            v_nl = v_nl * gates.view(1, -1, 1, 1)
            
            x = x + v_nl  # residual in latent
        
        # Warp + glitch
        x_warp = self._warp_latent(x)
        x_glitch = self._glitch_roll(x)
        x = x_warp * (1 - self.glitch_strength) + x_glitch * self.glitch_strength
        
        # Collapse back to sphere channels
        collapse_kernel = self.kernels[-1]
        u = F.conv2d(x, collapse_kernel)  # (1, sphere_dim, H, W)
        u = u.squeeze(0)
        
        # Tangent projection + sphere move
        v = self.tangent_project(x0, u)
        effective_step = self.step_size * residual_alpha
        
        if self.use_expmap:
            x_new = self.sphere_expmap(x0, v, step=effective_step)
        else:
            x_new = self.sphere_retract(x0, v, step=effective_step)
        
        x_new = self.sphere_normalize(x_new, dim=0)
        return x_new
    
    def get_soul_sliders(self):
        """Sliders for chaotic control."""
        return [
            {"label": "Step Size", "value_attr": "step_size", "min_value": 0.05, "max_value": 1.0},
            {"label": "Keep Ratio", "value_attr": "selection_ratio", "min_value": 0.05, "max_value": 0.9},
            {"label": "Chaos Intensity", "value_attr": "chaos_intensity", "min_value": 2.5, "max_value": 4.0},
            {"label": "Gate Sharpness", "value_attr": "gate_sharpness", "min_value": 2.0, "max_value": 30.0},
            {"label": "Glitch", "value_attr": "glitch_strength", "min_value": 0.0, "max_value": 1.0},
            {"label": "Warp", "value_attr": "warp_strength", "min_value": 0.0, "max_value": 0.6},
            {"label": "Pulse Mix", "value_attr": "pulse_mix", "min_value": 0.0, "max_value": 1.0},
        ]
