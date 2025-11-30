from main import ConvolutionArt
from souls import Soul, Seraphim, Cherubim, Pandemonium, Lenia, InceptionSoul, CryptoLenia

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

# Option 4: Lenia - Smooth-life waves on the sphere
lenia = Lenia(
    radii=(3, 7, 11),
    growth_peak=0.35,
    growth_width=0.12,
    growth_strength=1.4,
    feedback=0.35,
    jitter_strength=0.08,
    warp_roll=2,
    drift_magnitude=0.04,
    momentum=0.65,
    sphere_dim=10,
    step_size=0.35,
    use_expmap=True,
)

# Option 5: InceptionSoul - Inception blocks inside a U-Net scaffold
inception = InceptionSoul(
    depth=3,
    base_channels=16,
    bottleneck_blocks=2,
    drift_magnitude=0.006,
    momentum=0.7,
    sphere_dim=4,
    step_size=0.3,
    skip_strength=0.6,
    nonlinearity_strength=1.0,
    use_expmap=True,
)

# Option 6: CryptoLenia - Rotating anisotropic Lenia weirdness
cryptolenia = CryptoLenia(
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
    sphere_dim=4,
    step_size=0.32,
    use_expmap=True,
)

# Select which soul to use
soul = cryptolenia  # or lenia / pandemonium / cherubim / seraphim / inception

# Choose mapping method:
# - "orthonormal": Fast, stable, good for any dimension (recommended)
# - "hsv": More perceptually uniform colors
# - "adaptive_pca": Auto-adjusts to maximize visible variance
mapping_method = "adaptive_pca"

app = ConvolutionArt(
    conv_processor=soul, 
    bpm=60,
    mapping_method=mapping_method,
    use_compile=False  # Note: torch.compile slower on MPS, faster on CUDA
)
app.run()
