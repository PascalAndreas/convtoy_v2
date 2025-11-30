"""
Convolution processors for image manipulation.

This module provides a base class and concrete implementations for different
convolution-based image processing strategies.
"""

import torch
import numpy as np
from abc import ABC, abstractmethod


class Soul(ABC):
    """
    Base class for convolution-based image processors.
    
    Subclasses should implement the convolution logic and can customize
    kernel initialization, drift behavior, and layer composition.
    """
    
    def __init__(self, padding=3, drift_magnitude=0.02, momentum=0.7, sphere_dim=3, device=None):
        """
        Initialize the convolution processor.
        
        Args:
            padding: Padding size for image (kernel_size // 2, default: 3)
            drift_magnitude: Magnitude of drift direction vector
            momentum: Momentum factor for drift direction updates (0-1)
            sphere_dim: Dimensionality of the unit sphere (default: 3)
            device: PyTorch device (cuda, mps, or cpu)
        """
        self.padding = padding
        self.sphere_dim = sphere_dim
        
        # Device selection: prefer CUDA > MPS (Apple Silicon) > CPU
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device
        
        self.drift_magnitude = drift_magnitude
        self.momentum = momentum
        
        # Initialize kernels
        self.kernels = self._initialize_kernels()
        
        # Ensure kernels are zero-mean
        self.kernels = [self._enforce_zero_mean(k) for k in self.kernels]
        
        # Store initial kernel scales (RMS = root mean square) to maintain energy
        # For zero-mean signals, RMS is the natural measure of energy
        self.kernel_scales = [torch.sqrt(torch.mean(k ** 2)).item() for k in self.kernels]
        
        # Initialize drift directions (one per kernel, on unit sphere)
        self.drift_directions = []
        self.drift_velocities = []  # For momentum
        for kernel in self.kernels:
            # Initialize random direction on unit sphere
            direction = torch.randn_like(kernel)
            direction = self._enforce_zero_mean(direction)  # Ensure zero-mean
            direction = direction / (direction.norm() + 1e-8)
            self.drift_directions.append(direction)
            
            # Initialize velocity to zero
            velocity = torch.zeros_like(kernel)
            self.drift_velocities.append(velocity)
    
    @abstractmethod
    def _initialize_kernels(self):
        """Initialize convolution kernels."""
        pass
    
    @abstractmethod
    def apply(self, image, residual_alpha=0.2):
        """
        Apply convolution processing to an image.
        
        Args:
            image: Input image tensor (C, H, W)
            residual_alpha: Blending factor for residual connections

        Returns:
            Processed image tensor (C, H, W)
        """
        pass
    
    def get_base_sliders(self):
        """Return base sliders that all souls have."""
        return [{"label": "Momentum", "value_attr": "momentum"}]
    
    def get_soul_sliders(self):
        """Return soul-specific sliders. Override this in subclasses."""
        return []

    def get_sliders(self):
        """Return all slider definitions (base + soul-specific)."""
        return self.get_base_sliders() + self.get_soul_sliders()
    
    def _enforce_zero_mean(self, kernel):
        """Enforce zero-mean constraint on a kernel."""
        return kernel - kernel.mean()
    
    def randomize_kernels(self):
        """Randomize all kernels and reset drift directions."""
        self.kernels = self._initialize_kernels()
        
        # Ensure kernels are zero-mean
        self.kernels = [self._enforce_zero_mean(k) for k in self.kernels]
        
        # Reset kernel scales (RMS for zero-mean signals)
        self.kernel_scales = [torch.sqrt(torch.mean(k ** 2)).item() for k in self.kernels]
        
        # Reset drift directions and velocities
        self.drift_directions = []
        self.drift_velocities = []
        for kernel in self.kernels:
            direction = torch.randn_like(kernel)
            direction = self._enforce_zero_mean(direction)  # Ensure zero-mean
            direction = direction / (direction.norm() + 1e-8)
            self.drift_directions.append(direction)
            
            velocity = torch.zeros_like(kernel)
            self.drift_velocities.append(velocity)
    
    def change_drift(self, change_amount):
        """
        Update drift directions via random walk on unit sphere with momentum.
        
        This method updates the direction of drift for each kernel. The drift
        direction performs a random walk on an n-dimensional unit sphere, where
        n is the number of elements in each kernel.
        
        Args:
            change_amount: Amount of change to apply to drift direction
        """
        if change_amount > 0:
            for i in range(len(self.kernels)):
                # Generate random tangent perturbation
                tangent_noise = torch.randn_like(self.drift_directions[i]) * change_amount
                
                # Apply momentum to the velocity
                self.drift_velocities[i] = (self.momentum * self.drift_velocities[i] + 
                                           (1 - self.momentum) * tangent_noise)
                
                # Update direction with velocity
                new_direction = self.drift_directions[i] + self.drift_velocities[i]
                
                # Ensure zero-mean before projecting to unit sphere
                new_direction = self._enforce_zero_mean(new_direction)
                
                # Project back onto unit sphere
                new_direction = new_direction / (new_direction.norm() + 1e-8)
                
                self.drift_directions[i] = new_direction
    
    def apply_perturbation(self, image, mask, strength=0.1, mode='noise'):
        """
        Apply perturbation to sphere image using a mask, preserving sphere constraint.
        
        This is device-accelerated and respects the unit sphere constraint.
        
        Args:
            image: Image tensor (C, H, W) on sphere, on any device
            mask: Perturbation mask (H, W) on any device, values 0-1
            strength: Perturbation strength
            mode: 'noise' or 'swirl' (default: 'noise')
            
        Returns:
            Perturbed image tensor, still on unit sphere
        """
        if strength <= 0:
            return image
        
        # Ensure tensors are on the same device
        image = image.to(self.device)
        mask = mask.to(self.device)
        
        if mode == 'noise':
            # Random tangent perturbation on sphere
            # Generate random noise
            noise = torch.randn_like(image) * strength
            
            # Project to tangent space
            tangent_noise = self.tangent_project(image, noise)
            
            # Apply mask
            tangent_perturbation = tangent_noise * mask.unsqueeze(0)
            
            # Move on sphere using retraction
            return self.sphere_retract(image, tangent_perturbation, step=1.0)
        
        elif mode == 'swirl':
            # Swirl/rotation perturbation (more interesting visual effect)
            # Create small rotation field where mask is high
            angle = mask * strength * 0.5  # Rotation angle field
            
            # Find center of mass of the mask (where user clicked)
            h, w = image.shape[1], image.shape[2]
            y, x = torch.meshgrid(torch.arange(h, device=self.device, dtype=torch.float32), 
                                 torch.arange(w, device=self.device, dtype=torch.float32), indexing='ij')
            
            # Center of perturbation (weighted by mask)
            mask_sum = mask.sum() + 1e-8
            center_y = (y * mask).sum() / mask_sum
            center_x = (x * mask).sum() / mask_sum
            
            dy, dx = y - center_y, x - center_x
            
            # Apply rotation based on mask strength
            cos_a = torch.cos(angle)
            sin_a = torch.sin(angle)
            
            new_y = (dy * cos_a - dx * sin_a + center_y).long().clamp(0, h - 1)
            new_x = (dy * sin_a + dx * cos_a + center_x).long().clamp(0, w - 1)
            
            swirled = image[:, new_y, new_x]
            
            # Blend swirled with original based on mask in tangent space
            # This creates a smooth transition that stays on sphere
            diff = swirled - image
            tangent_diff = self.tangent_project(image, diff)
            
            # Apply mask to tangent velocity
            alpha = mask.unsqueeze(0) * 0.5
            tangent_velocity = tangent_diff * alpha
            
            # Move on sphere
            return self.sphere_retract(image, tangent_velocity, step=1.0)
        
        return image
    
    def apply_drift(self, heart_signal):
        """
        Apply drift to kernels using the heart signal and current drift directions.
        
        From first principles:
        - Drift must preserve zero-mean (done by having zero-mean drift directions)
        - Kernel energy must stay bounded to prevent explosion or collapse
        - But we want soft constraints that allow organic evolution
        
        Args:
            heart_signal: Signal value from the heart (typically from ECG)
        """
        if abs(heart_signal) > 1e-8:
            for i in range(len(self.kernels)):
                # Apply drift in the current direction (zero-mean by construction)
                drift = self.drift_directions[i] * self.drift_magnitude * heart_signal
                self.kernels[i] = self.kernels[i] + drift
                
                # Maintain energy: use RMS (root mean square) for zero-mean signals
                # RMS is more principled than std for maintaining signal energy
                current_rms = torch.sqrt(torch.mean(self.kernels[i] ** 2))
                target_rms = self.kernel_scales[i]
                
                if current_rms > 1e-8:
                    # Soft pull toward target RMS with exponential averaging
                    # This allows drift while preventing unbounded growth/decay
                    target_scale = target_rms / current_rms
                    scale_factor = 0.9 + 0.1 * target_scale  # 10% correction toward target
                    self.kernels[i] = self.kernels[i] * scale_factor
    
    def sphere_normalize(self, x, dim=1, eps=1e-6):
        """
        Normalize vectors to unit sphere.
        
        Args:
            x: Tensor to normalize
            dim: Dimension along which to normalize (default: 1 for channel dimension)
            eps: Small epsilon to prevent division by zero
            
        Returns:
            Normalized tensor with ||x||_2 = 1 along specified dimension
        """
        return torch.nn.functional.normalize(x, dim=dim, eps=eps)
    
    def tangent_project(self, x, u):
        """
        Project vector u onto tangent space at point x on the sphere.
        
        For a point x on S^{n-1}, the tangent space is:
        T_x S^{n-1} = {v : v·x = 0}
        
        Projection: v = u - (u·x)x
        
        Args:
            x: Point on sphere (B, C, H, W) or (C, H, W) with ||x|| = 1
            u: Ambient vector to project
            
        Returns:
            Tangent vector v orthogonal to x
        """
        # Compute dot product along channel dimension
        # Use fused multiply-add for efficiency
        dot = (u * x).sum(dim=1 if x.dim() == 4 else 0, keepdim=True)
        # v = u - dot * x  (fused operation)
        v = torch.addcmul(u, dot, x, value=-1.0)
        return v
    
    def sphere_retract(self, x, v, step=1.0, eps=1e-6):
        """
        Retraction: move from x in direction v and project back to sphere.
        
        Simple retraction (cheap and effective):
        x_new = normalize(x + step * v)
        
        Args:
            x: Current point on sphere (C, H, W) or (B, C, H, W)
            v: Tangent vector (same shape as x)
            step: Step size (default: 1.0)
            eps: Small epsilon for numerical stability
            
        Returns:
            New point on sphere
        """
        x_new = x + step * v
        return self.sphere_normalize(x_new, dim=1 if x_new.dim() == 4 else 0, eps=eps)
    
    def sphere_expmap(self, x, v, step=1.0, eps=1e-6):
        """
        Exponential map: geodesic on sphere from x in direction v.
        
        If v is tangent at x, then:
        x_new = cos(step * ||v||) * x + sin(step * ||v||) * (v / ||v||)
        
        This is the exact geodesic on the sphere.
        
        Args:
            x: Current point on sphere (C, H, W) or (B, C, H, W)
            v: Tangent vector (same shape as x)
            step: Step size (default: 1.0)
            eps: Small epsilon for numerical stability
            
        Returns:
            New point on sphere following geodesic
        """
        # Compute norm of v along channel dimension
        # Use p=2 and keepdim for efficiency
        vnorm = torch.linalg.vector_norm(v, dim=1 if v.dim() == 4 else 0, keepdim=True).clamp_min(eps)
        
        # Compute scaled angle
        angle = step * vnorm
        
        # Exponential map formula using fused operations
        # Reuse vnorm to avoid division
        v_normalized = v / vnorm
        x_new = torch.addcmul(torch.cos(angle) * x, torch.sin(angle), v_normalized)
        
        return x_new
