from main import ConvolutionArt
from souls import Soul, Seraphim, Cherubim, Pandemonium

# ===== Choose Your Soul =====

# Option 1: Seraphim - Sequential multi-layer convolutions
seraphim = Seraphim(
    kernel_size=5,
    num_layers=20,
    drift_magnitude=0.02, 
    momentum=0.7,
    sphere_dim=4,
    step_size=0.3,
    use_expmap=False
)

# Option 2: Cherubim - Encoder-decoder with U-Net skip connections
# More interesting multi-scale behaviors!
cherubim = Cherubim(
    depth=3,                    # Number of encoder/decoder levels
    latent_dim=12,              # Channels at bottleneck (more = richer)
    drift_magnitude=0.02,
    momentum=0.7,
    sphere_dim=4,               # Start dimension
    step_size=0.3,
    use_expmap=False,
    skip_strength=0.5,          # U-Net skip connections (adjustable)
    nonlinearity_strength=1.0,  # Tanh saturation (adjustable)
    bottleneck_iterations=4     # Extra processing at lowest resolution
)

# Option 3: Pandemonium - Chaotic gating, warps, glitches, and pruning
pandemonium = Pandemonium(
    latent_dim=36,
    num_blocks=5,
    drift_magnitude=0.015,
    momentum=0.6,
    sphere_dim=4,
    step_size=0.4,
    selection_ratio=0.45,
    chaos_intensity=3.8,
    gate_sharpness=14.0,
    glitch_strength=0.35,
    warp_strength=0.25,
    pulse_mix=0.6,
    use_expmap=False,
)

# Select which soul to use
soul = pandemonium  # or cherubim / seraphim

# Choose mapping method:
# - "orthonormal": Fast, stable, good for any dimension (recommended)
# - "hsv": More perceptually uniform colors
# - "adaptive_pca": Auto-adjusts to maximize visible variance
mapping_method = "orthonormal"

app = ConvolutionArt(
    conv_processor=soul, 
    bpm=60,
    mapping_method=mapping_method,
    use_compile=False  # Note: torch.compile slower on MPS, faster on CUDA
)
app.run()
