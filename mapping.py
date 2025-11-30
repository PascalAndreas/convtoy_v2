"""
Sphere-to-RGB mapping strategies for visualizing n-dimensional spheres.

This module provides various approaches for projecting high-dimensional
unit sphere states to RGB colors for visualization.
"""

import torch
import torch.nn.functional as F


class SphereMapper:
    """Base class for sphere-to-RGB mapping strategies."""
    
    def __init__(self, sphere_dim, device="cpu", seed=0):
        """
        Initialize sphere mapper.
        
        Args:
            sphere_dim: Dimensionality of the unit sphere
            device: PyTorch device
            seed: Random seed for reproducible mappings
        """
        self.sphere_dim = sphere_dim
        self.device = device
        self.seed = seed
    
    def map_to_rgb(self, sphere_img):
        """
        Map sphere image to RGB.
        
        Args:
            sphere_img: Tensor (sphere_dim, H, W) or (B, sphere_dim, H, W)
            
        Returns:
            RGB tensor (3, H, W) or (B, 3, H, W) with values in [0, 1]
        """
        raise NotImplementedError


class OrthonormalRGBMapper(SphereMapper):
    """
    Project onto fixed orthonormal RGB basis.
    
    This is the "first principles" approach: pick three orthonormal
    vectors in R^n and use them as RGB axes. Fast, stable, and works
    for any dimension.
    
    Advantages:
    - Constant-time regardless of dimension
    - Deterministic (seedable)
    - Preserves sphere structure
    - No special cases needed
    """
    
    def __init__(self, sphere_dim, device="cpu", seed=0):
        super().__init__(sphere_dim, device, seed)
        self.rgb_basis = self._make_rgb_basis()
    
    def _make_rgb_basis(self):
        """
        Create orthonormal basis for RGB projection using Gram-Schmidt.
        
        Returns:
            Tensor of shape (3, sphere_dim) with orthonormal rows
        """
        g = torch.Generator(device=self.device)
        g.manual_seed(self.seed)
        
        # Generate random vectors
        U = torch.randn(3, self.sphere_dim, generator=g, device=self.device)
        
        # Gram-Schmidt orthonormalization
        # First vector
        U[0] = F.normalize(U[0], dim=0, eps=1e-8)
        
        # Second vector (orthogonal to first)
        U[1] = U[1] - (U[1] @ U[0]) * U[0]
        U[1] = F.normalize(U[1], dim=0, eps=1e-8)
        
        # Third vector (orthogonal to first two)
        U[2] = U[2] - (U[2] @ U[0]) * U[0] - (U[2] @ U[1]) * U[1]
        U[2] = F.normalize(U[2], dim=0, eps=1e-8)
        
        return U
    
    def map_to_rgb(self, sphere_img):
        """
        Project sphere image onto RGB basis.
        
        Args:
            sphere_img: Tensor (sphere_dim, H, W) with values in [-1, 1]
            
        Returns:
            RGB tensor (3, H, W) with values in [0, 1]
        """
        # sphere_img should already be on device - avoid unnecessary transfer
        
        # Use matrix multiply instead of einsum (faster on most hardware)
        # Reshape: (n, H, W) -> (n, H*W)
        if sphere_img.dim() == 3:
            n, h, w = sphere_img.shape
            sphere_flat = sphere_img.reshape(n, -1)
            
            # Matrix multiply: (3, n) @ (n, H*W) -> (3, H*W)
            rgb_flat = self.rgb_basis @ sphere_flat
            
            # Reshape back: (3, H*W) -> (3, H, W)
            rgb = rgb_flat.reshape(3, h, w)
        elif sphere_img.dim() == 4:
            b, n, h, w = sphere_img.shape
            sphere_flat = sphere_img.reshape(b, n, -1)
            
            # Batch matrix multiply
            rgb_flat = torch.bmm(
                self.rgb_basis.unsqueeze(0).expand(b, -1, -1),
                sphere_flat
            )
            rgb = rgb_flat.reshape(b, 3, h, w)
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got {sphere_img.dim()}D")
        
        # Fast scale and clamp: [-1, 1] -> [0, 1]
        # rgb_out = (rgb + 1) * 0.5 = rgb * 0.5 + 0.5
        rgb = rgb.mul_(0.5).add_(0.5)  # in-place operations
        rgb = rgb.clamp_(0.0, 1.0)  # in-place clamp
        
        return rgb
    
    def update_basis(self, new_seed=None):
        """
        Regenerate the RGB basis with a new seed.
        
        Args:
            new_seed: New random seed (if None, uses self.seed + 1)
        """
        if new_seed is not None:
            self.seed = new_seed
        else:
            self.seed += 1
        self.rgb_basis = self._make_rgb_basis()


class HSVMapper(SphereMapper):
    """
    Map to HSV color space for more perceptually uniform colors.
    
    Uses three orthonormal projections:
    - Two dimensions for hue (angle in 2D plane)
    - One dimension for value/brightness
    - Saturation is constant (or based on dynamics)
    
    Advantages:
    - More visually distinct colors
    - Better for seeing structure
    - Hue wraps around naturally
    """
    
    def __init__(self, sphere_dim, device="cpu", seed=0, saturation=0.8):
        super().__init__(sphere_dim, device, seed)
        self.saturation = saturation
        self.basis = self._make_basis()
    
    def _make_basis(self):
        """Create orthonormal basis (3 vectors) for HSV mapping."""
        g = torch.Generator(device=self.device)
        g.manual_seed(self.seed)
        
        U = torch.randn(3, self.sphere_dim, generator=g, device=self.device)
        
        # Gram-Schmidt
        U[0] = F.normalize(U[0], dim=0, eps=1e-8)
        U[1] = U[1] - (U[1] @ U[0]) * U[0]
        U[1] = F.normalize(U[1], dim=0, eps=1e-8)
        U[2] = U[2] - (U[2] @ U[0]) * U[0] - (U[2] @ U[1]) * U[1]
        U[2] = F.normalize(U[2], dim=0, eps=1e-8)
        
        return U
    
    def hsv_to_rgb_torch(self, h, s, v):
        """
        Convert HSV to RGB using PyTorch (vectorized, optimized).
        
        Uses continuous functions instead of branching for better performance.
        
        Args:
            h: Hue in [0, 1]
            s: Saturation in [0, 1]
            v: Value in [0, 1]
            
        Returns:
            RGB tensor with values in [0, 1]
        """
        # Vectorized HSV to RGB without branching (much faster!)
        # Based on standard HSV formula but optimized
        
        h = h * 6.0  # Scale to [0, 6]
        
        # Compute RGB channels using smooth functions
        # r = v * (1 - s * max(0, min(1, min(k, 4-k))))
        # where k varies for each channel
        
        def channel(offset):
            k = (offset + h) % 6.0
            # max(0, min(k, 4-k, 1))
            t = torch.clamp(torch.minimum(k, 4.0 - k), 0.0, 1.0)
            return v * (1.0 - s * t)
        
        r = channel(5.0)
        g = channel(3.0)
        b = channel(1.0)
        
        # Stack into RGB format
        if h.dim() == 2:
            # (H, W) -> (3, H, W)
            return torch.stack([r, g, b], dim=0)
        else:
            # (B, H, W) -> (B, 3, H, W)
            return torch.stack([r, g, b], dim=1)
    
    def map_to_rgb(self, sphere_img):
        """
        Map sphere image to RGB via HSV.
        
        Args:
            sphere_img: Tensor (sphere_dim, H, W) with values in [-1, 1]
            
        Returns:
            RGB tensor (3, H, W) with values in [0, 1]
        """
        # Use matrix multiply instead of einsum (faster)
        if sphere_img.dim() == 3:
            n, h, w = sphere_img.shape
            sphere_flat = sphere_img.reshape(n, -1)
            
            # Matrix multiply: (3, n) @ (n, H*W) -> (3, H*W)
            proj_flat = self.basis @ sphere_flat
            proj = proj_flat.reshape(3, h, w)
            
            p0, p1, p2 = proj[0], proj[1], proj[2]
        else:
            batch_size, n, h, w = sphere_img.shape
            sphere_flat = sphere_img.reshape(batch_size, n, -1)
            
            proj_flat = torch.bmm(
                self.basis.unsqueeze(0).expand(batch_size, -1, -1),
                sphere_flat
            )
            proj = proj_flat.reshape(batch_size, 3, h, w)
            
            p0, p1, p2 = proj[:, 0], proj[:, 1], proj[:, 2]
        
        # Hue from angle in (p0, p1) plane
        hue = torch.atan2(p1, p0)
        hue = (hue * 0.1591549430918953 + 0.5) % 1.0  # / (2*pi) + 0.5, then wrap
        
        # Value from third dimension - fused operation
        value = p2 * 0.5 + 0.5  # Map [-1, 1] to [0, 1]
        
        # Constant saturation - no need for full_like, broadcast instead
        saturation = self.saturation
        
        # Convert HSV to RGB
        rgb = self.hsv_to_rgb_torch(hue, saturation, value)
        
        return rgb.clamp_(0, 1)  # in-place clamp


class AdaptivePCAMapper(SphereMapper):
    """
    Online PCA: adapt RGB basis to maximize visible variance.
    
    Maintains running estimate of top-3 principal directions.
    This automatically finds the "interesting" subspace and maximizes
    visual contrast.
    
    Advantages:
    - Auto-contrast: always shows maximum variation
    - Adapts to dynamics
    - Great for finding hidden structure
    
    Disadvantages:
    - Colors change over time (not stable)
    - Slightly more computation
    """
    
    def __init__(self, sphere_dim, device="cpu", learning_rate=0.01, seed=0):
        super().__init__(sphere_dim, device, seed)
        self.learning_rate = learning_rate
        
        # Initialize random orthonormal basis
        self.rgb_basis = self._initialize_basis()
    
    def _initialize_basis(self):
        """Initialize random orthonormal basis."""
        g = torch.Generator(device=self.device)
        g.manual_seed(self.seed)
        
        U = torch.randn(3, self.sphere_dim, generator=g, device=self.device)
        
        # Gram-Schmidt
        U[0] = F.normalize(U[0], dim=0, eps=1e-8)
        U[1] = U[1] - (U[1] @ U[0]) * U[0]
        U[1] = F.normalize(U[1], dim=0, eps=1e-8)
        U[2] = U[2] - (U[2] @ U[0]) * U[0] - (U[2] @ U[1]) * U[1]
        U[2] = F.normalize(U[2], dim=0, eps=1e-8)
        
        return U
    
    def update_basis(self, sphere_img):
        """
        Update basis using Oja's rule (online PCA).
        
        For each principal direction u_i:
        u_i ← u_i + η * x * (x · u_i)
        u_i ← normalize(u_i)
        
        Then orthogonalize lower directions.
        
        Args:
            sphere_img: Current sphere state (sphere_dim, H, W)
        """
        # Flatten spatial dimensions: (sphere_dim, H*W)
        if sphere_img.dim() == 3:
            x_flat = sphere_img.reshape(self.sphere_dim, -1)
        else:
            # (B, C, H, W) -> average over batch and flatten
            x_flat = sphere_img.mean(dim=0).reshape(self.sphere_dim, -1)
        
        # Sample random pixels for efficiency (online update)
        n_samples = min(100, x_flat.shape[1])
        indices = torch.randperm(x_flat.shape[1], device=self.device)[:n_samples]
        x_samples = x_flat[:, indices]  # (sphere_dim, n_samples)
        
        # Update each basis vector
        for i in range(3):
            u_i = self.rgb_basis[i]  # (sphere_dim,)
            
            # Compute dot products with samples
            dots = x_samples.T @ u_i  # (n_samples,)
            
            # Oja update: u += η * x * (x·u)
            update = (x_samples @ dots) / n_samples
            u_i = u_i + self.learning_rate * update
            
            # Normalize
            u_i = F.normalize(u_i, dim=0, eps=1e-8)
            
            self.rgb_basis[i] = u_i
        
        # Re-orthogonalize (Gram-Schmidt)
        self.rgb_basis[1] = self.rgb_basis[1] - (self.rgb_basis[1] @ self.rgb_basis[0]) * self.rgb_basis[0]
        self.rgb_basis[1] = F.normalize(self.rgb_basis[1], dim=0, eps=1e-8)
        
        self.rgb_basis[2] = (self.rgb_basis[2] - 
                            (self.rgb_basis[2] @ self.rgb_basis[0]) * self.rgb_basis[0] -
                            (self.rgb_basis[2] @ self.rgb_basis[1]) * self.rgb_basis[1])
        self.rgb_basis[2] = F.normalize(self.rgb_basis[2], dim=0, eps=1e-8)
    
    def map_to_rgb(self, sphere_img, update=True):
        """
        Map sphere image to RGB with optional basis update.
        
        Args:
            sphere_img: Tensor (sphere_dim, H, W)
            update: Whether to update the basis (online learning)
            
        Returns:
            RGB tensor (3, H, W) with values in [0, 1]
        """
        # Update basis if requested
        if update:
            self.update_basis(sphere_img)
        
        # Use matrix multiply instead of einsum (faster)
        if sphere_img.dim() == 3:
            n, h, w = sphere_img.shape
            sphere_flat = sphere_img.reshape(n, -1)
            rgb_flat = self.rgb_basis @ sphere_flat
            rgb = rgb_flat.reshape(3, h, w)
        else:
            b, n, h, w = sphere_img.shape
            sphere_flat = sphere_img.reshape(b, n, -1)
            rgb_flat = torch.bmm(
                self.rgb_basis.unsqueeze(0).expand(b, -1, -1),
                sphere_flat
            )
            rgb = rgb_flat.reshape(b, 3, h, w)
        
        # Fast scale and clamp: [-1, 1] -> [0, 1]
        rgb = rgb.mul_(0.5).add_(0.5)  # in-place operations
        rgb = rgb.clamp_(0.0, 1.0)
        
        return rgb


def create_mapper(method="orthonormal", sphere_dim=3, device="cpu", seed=0, **kwargs):
    """
    Factory function to create sphere mappers.
    
    Args:
        method: Mapping method ("orthonormal", "hsv", "adaptive_pca")
        sphere_dim: Dimensionality of sphere
        device: PyTorch device
        seed: Random seed
        **kwargs: Additional arguments for specific mappers
        
    Returns:
        SphereMapper instance
    """
    if method == "orthonormal":
        return OrthonormalRGBMapper(sphere_dim, device, seed)
    elif method == "hsv":
        saturation = kwargs.get("saturation", 0.8)
        return HSVMapper(sphere_dim, device, seed, saturation)
    elif method == "adaptive_pca":
        learning_rate = kwargs.get("learning_rate", 0.01)
        return AdaptivePCAMapper(sphere_dim, device, learning_rate, seed)
    else:
        raise ValueError(f"Unknown mapping method: {method}")

