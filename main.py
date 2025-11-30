import pygame
import torch
import torch.nn.functional as F
import numpy as np
import sys
import time
import os
from datetime import datetime
from collections import deque
from souls import Seraphim
from heart import Heart
from mapping import create_mapper

# Initialize Pygame
pygame.init()

# Constants
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (100, 100, 100)
BUTTON_COLOR = (70, 70, 70)
BUTTON_HOVER = (90, 90, 90)
TEXT_COLOR = (255, 255, 255)

# UI dimensions
UI_PANEL_WIDTH = 250  # Width of right panel with buttons/sliders
IMAGE_MARGIN = 20  # Margin around image
INSTRUCTIONS_HEIGHT = 220  # Height needed for instructions at bottom

class ConvolutionArt:
    def __init__(self, conv_processor=None, bpm=120, mapping_method="orthonormal", use_compile=True):
        # Device (use CUDA > MPS > CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Convolution processor
        if conv_processor is None:
            conv_processor = Seraphim(kernel_size=7, num_layers=5, device=self.device)
        self.conv_processor = conv_processor
        
        # Optimize with torch.compile if requested and available (PyTorch 2.0+)
        self.use_compile = use_compile
        if use_compile and hasattr(torch, 'compile') and self.device.type != 'mps':
            try:
                print("Compiling model with torch.compile...")
                self.conv_processor.apply = torch.compile(
                    self.conv_processor.apply,
                    mode="reduce-overhead"  # Best for repeated calls
                )
                print("✓ Model compiled successfully")
            except Exception as e:
                print(f"Warning: torch.compile failed ({e}), continuing without compilation")
                self.use_compile = False
        
        # Heart simulator for driving drift
        # Sample rate matches FPS, amplitude will be scaled by drift_scale
        # HRV adds natural beat-to-beat variation (5% variability, 15 breaths/min)
        self.heart = Heart(bpm=bpm, sample_rate=FPS, amplitude=1.0,
                          hrv_amount=0.05, breathing_rate=0.25)
        
        # Drift scaling factor (multiplies ECG signal)
        self.drift_scale = 0.8
        
        # Sphere dimension (for n-dimensional unit sphere)
        self.sphere_dim = self.conv_processor.sphere_dim
        
        # Sphere-to-RGB mapper
        self.mapper = create_mapper(
            method=mapping_method,
            sphere_dim=self.sphere_dim,
            device=self.device,
            seed=42
        )
        self.mapping_method = mapping_method
        print(f"Using {mapping_method} mapping for {self.sphere_dim}D sphere")
        
        # Compile the mapper for better performance (skip on MPS where it's slower)
        if use_compile and hasattr(torch, 'compile') and self.device.type != 'mps':
            try:
                self.mapper.map_to_rgb = torch.compile(
                    self.mapper.map_to_rgb,
                    mode="reduce-overhead"
                )
                print("✓ Mapper compiled successfully")
            except Exception as e:
                print(f"Warning: mapper compile failed ({e}), continuing without compilation")
        
        # Image dimensions (no scaling needed)
        # Base size - 2*padding to account for circular padding during convolution
        self.img_width = 1024
        self.img_height = 1024
        
        # Calculate window size dynamically based on image size
        self.window_width = self.img_width + UI_PANEL_WIDTH + IMAGE_MARGIN * 3
        self.window_height = max(self.img_height + IMAGE_MARGIN * 2, INSTRUCTIONS_HEIGHT + 200)
        
        # Create screen with calculated dimensions
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Convolution Art Toy")
        self.clock = pygame.time.Clock()
        self.running = True
        
        # Fullscreen mode
        self.fullscreen = False
        self.original_size = (self.window_width, self.window_height)
        self.original_img_size = (self.img_width, self.img_height)
        
        # Initialize random image
        self.image = self._random_image()
        
        # Image noise settings (continuous center + vignette perturbation)
        self.noise_amount = 0.0  # How much random noise to add to image each frame
        
        # Residual/blend parameter (prevents collapse)
        self.residual_alpha = 0.8  # How much of new conv to blend in (0=no change, 1=full replacement)
        
        # Precompute center + vignette mask (on device for efficiency)
        self.perturbation_mask = self._create_perturbation_mask()
        
        # Mouse perturbation settings
        self.mouse_pressed = False
        self.mouse_press_start = 0
        self.mouse_pos = (0, 0)
        self.perturb_radius = 80  # Increased for larger resolution
        self.display_offset = (IMAGE_MARGIN, IMAGE_MARGIN)
        
        # UI elements
        self.font = pygame.font.Font(None, 24)
        self.buttons = self._create_buttons()
        self.sliders = self._create_sliders()
        
        # Create reusable surface for efficient pixel updates (32-bit RGBA for full color)
        self.display_surface = pygame.Surface((self.img_width, self.img_height), depth=32)
        
        # Pre-allocate buffers to avoid allocation overhead each frame
        # RGB buffer (reused each frame to avoid allocations)
        self.rgb_buffer = torch.zeros(
            (3, self.img_height, self.img_width),
            dtype=torch.float32,
            device=self.device
        )
        
        # uint8 buffer for CPU transfer
        self.uint8_buffer = torch.zeros(
            (3, self.img_height, self.img_width),
            dtype=torch.uint8,
            device=self.device
        )
        
        print(f"✓ Pre-allocated render buffers on {self.device}")
        
        # Performance tracking
        self.show_timing = False  # Toggle with 'T' key
        self.timing_history = {
            'drift': deque(maxlen=60),
            'noise': deque(maxlen=60),
            'conv': deque(maxlen=60),
            'render_map': deque(maxlen=60),       # Sphere to RGB mapping
            'render_sync': deque(maxlen=60),      # GPU sync + uint8 convert
            'render_cpu': deque(maxlen=60),       # GPU->CPU transfer
            'render_numpy': deque(maxlen=60),     # Tensor to numpy
            'render_blit': deque(maxlen=60),      # Pygame blit
            'render_ui': deque(maxlen=60),        # UI drawing
            'render_flip': deque(maxlen=60),      # Display flip
            'render_total': deque(maxlen=60),     # Total render
            'total': deque(maxlen=60)
        }
        
        # Playback control
        self.paused = False
        
        # Screenshot directory
        self.screenshot_dir = "screenshots"
        os.makedirs(self.screenshot_dir, exist_ok=True)
        
    def _random_image(self):
        """
        Generate a random image with pixels on the unit sphere.
        
        Each pixel is a point on S^{sphere_dim-1}, initialized as normalized Gaussian.
        This gives a uniform-like distribution on the sphere.
        
        Returns:
            Tensor of shape (sphere_dim, H, W) with ||img[:, h, w]|| = 1
        """
        # Generate Gaussian noise
        img = torch.randn(self.sphere_dim, self.img_height, self.img_width, 
                         device=self.device)
        
        # Normalize each pixel to unit sphere
        img = F.normalize(img, dim=0, eps=1e-6)
        
        return img
    
    def _sphere_to_rgb(self, sphere_img):
        """
        Map n-dimensional sphere points to RGB for rendering.
        
        Uses the configured mapper (orthonormal projection, HSV, or adaptive PCA).
        
        Args:
            sphere_img: Tensor (sphere_dim, H, W) with values in [-1, 1]
            
        Returns:
            RGB tensor (3, H, W) with values in [0, 1]
        """
        # Use the mapper
        if self.mapping_method == "adaptive_pca":
            # Adaptive PCA updates its basis each frame
            return self.mapper.map_to_rgb(sphere_img, update=True)
        else:
            # Other mappers use fixed basis
            return self.mapper.map_to_rgb(sphere_img)
    
    def _create_perturbation_mask(self):
        """Create a mask for center + vignette perturbation (on device)"""
        y_coords, x_coords = torch.meshgrid(
            torch.arange(self.img_height, dtype=torch.float32, device=self.device),
            torch.arange(self.img_width, dtype=torch.float32, device=self.device),
            indexing='ij'
        )
        
        # Center coordinates
        center_x = self.img_width / 2.0
        center_y = self.img_height / 2.0
        
        # Distance from center (normalized)
        dx = (x_coords - center_x) / center_x
        dy = (y_coords - center_y) / center_y
        dist_from_center = torch.sqrt(dx**2 + dy**2)
        
        # Create center bump (Gaussian)
        center_mask = torch.exp(-(dist_from_center**2) / 0.5)
        
        # Create vignette (stronger near edges)
        vignette_mask = torch.clamp(dist_from_center - 0.3, 0, 1)
        
        # Combine: center bump + edge vignette
        mask = center_mask + vignette_mask * 0.5
        
        # Normalize to [0, 1] and keep on device
        return mask / mask.max()
    
    def _create_buttons(self):
        """Create UI buttons"""
        button_configs = [
            {"text": "Randomize Image (I)", "action": "randomize_image"},
            {"text": "Randomize Kernels (K)", "action": "randomize_kernel"},
            {"text": "Change Colors (C)", "action": "change_colors"}
        ]
        
        button_width = 180
        button_height = 40
        spacing = 10
        start_x = self.window_width - button_width - 20
        start_y = 20
        
        buttons = []
        for i, config in enumerate(button_configs):
            buttons.append({
                "rect": pygame.Rect(start_x, start_y + i * (button_height + spacing), 
                                   button_width, button_height),
                "text": config["text"],
                "action": config["action"],
                "hovered": False
            })
        return buttons
    
    def _create_sliders(self):
        """Create control sliders dynamically"""
        # Base slider definitions (main app sliders)
        slider_configs = [
            {"label": "Drift", "value_attr": "drift_scale", "target_obj": self},
            {"label": "Perturbation", "value_attr": "noise_amount", "max_value": 0.2, "target_obj": self},
            {"label": "Residual Mix", "value_attr": "residual_alpha", "target_obj": self}
        ]
        
        # Add soul-specific sliders
        for soul_slider in self.conv_processor.get_sliders():
            slider_configs.append({**soul_slider, "target_obj": self.conv_processor})
        
        # Build slider objects with dynamic layout
        slider_width = 180
        slider_height = 20
        slider_x = self.window_width - slider_width - 20
        slider_y = 140
        spacing = 60
        
        sliders = []
        for i, config in enumerate(slider_configs):
            target = config["target_obj"]
            value = getattr(target, config["value_attr"])
            
            # Apply defaults: min=0.0, max=1.0
            min_val = config.get("min_value", 0.0)
            max_val = config.get("max_value", 1.0)
            
            # Calculate handle position based on value range
            value_range = max_val - min_val
            normalized_pos = (value - min_val) / value_range if value_range > 0 else 0
            
            sliders.append({
                "rect": pygame.Rect(slider_x, slider_y + i * spacing, slider_width, slider_height),
                "handle_x": slider_x + int(slider_width * normalized_pos),
                "dragging": False,
                "label": config["label"],
                "value_attr": config["value_attr"],
                "min_value": min_val,
                "max_value": max_val,
                "target_obj": target
            })
        
        return sliders
    
    def apply_image_noise(self):
        """Add random noise with center + vignette pattern (device-accelerated)"""
        if self.noise_amount <= 0:
            return
        
        # Use device-accelerated perturbation
        self.image = self.conv_processor.apply_perturbation(
            self.image, 
            self.perturbation_mask, 
            strength=self.noise_amount,
            mode='noise'
        )
    
    def apply_mouse_perturbation(self):
        """Apply localized perturbation where mouse is pressed (device-accelerated)"""
        if not self.mouse_pressed:
            return
        
        # Check if mouse is over the image display area
        mouse_x, mouse_y = self.mouse_pos
        img_rect = pygame.Rect(self.display_offset[0], self.display_offset[1], 
                               self.img_width, self.img_height)
        if not img_rect.collidepoint(mouse_x, mouse_y):
            return
        
        # Convert to image coordinates (direct pixel mapping)
        img_x = int(mouse_x - self.display_offset[0])
        img_y = int(mouse_y - self.display_offset[1])
        
        # Calculate strength based on press duration
        press_duration = pygame.time.get_ticks() - self.mouse_press_start
        strength = min(press_duration / 1000.0, 3.0) * 0.08
        
        # Create coordinate grid on device
        y_coords, x_coords = torch.meshgrid(
            torch.arange(self.img_height, dtype=torch.float32, device=self.device),
            torch.arange(self.img_width, dtype=torch.float32, device=self.device),
            indexing='ij'
        )
        
        # Calculate toroidal distance (wrapping at edges)
        dx = torch.minimum(torch.abs(x_coords - img_x), self.img_width - torch.abs(x_coords - img_x))
        dy = torch.minimum(torch.abs(y_coords - img_y), self.img_height - torch.abs(y_coords - img_y))
        dist = torch.sqrt(dx**2 + dy**2)
        
        # Create localized Gaussian mask
        mask = torch.exp(-(dist**2) / (2 * self.perturb_radius**2))
        mask = torch.where(dist <= self.perturb_radius * 2, mask, torch.zeros_like(mask))
        
        # Apply perturbation using device-accelerated method
        self.image = self.conv_processor.apply_perturbation(
            self.image,
            mask,
            strength=strength,
            mode='swirl'  # Use swirl effect for more interesting visual
        )
    
    def image_to_surface(self, img_tensor):
        """
        Convert sphere tensor to pygame surface (copies from device to CPU for rendering).
        
        Args:
            img_tensor: Sphere image (sphere_dim, H, W)
            
        Returns:
            Pygame surface for display, and timing dict
        """
        timings = {}
        
        # Map sphere to RGB (stays on device)
        t0 = time.perf_counter()
        rgb_img = self._sphere_to_rgb(img_tensor)
        timings['map'] = time.perf_counter() - t0
        
        # Convert to uint8 - simple approach
        t0 = time.perf_counter()
        rgb_uint8 = (rgb_img * 255.0).to(torch.uint8)
        
        # CRITICAL: Synchronize GPU before transfer
        # This ensures GPU work completes BEFORE we measure transfer time
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        elif self.device.type == 'mps':
            torch.mps.synchronize()
        timings['sync'] = time.perf_counter() - t0
        
        # Copy to CPU for rendering (direct transfer, no pinned memory on MPS)
        t0 = time.perf_counter()
        img_cpu = rgb_uint8.cpu()
        timings['cpu'] = time.perf_counter() - t0
        
        # Convert to numpy (no copy needed, shares memory)
        # Permute to (H, W, 3) for pygame
        t0 = time.perf_counter()
        img_np = img_cpu.permute(1, 2, 0).numpy()
        timings['numpy'] = time.perf_counter() - t0
        
        # Blit to surface (swapaxes for pygame's expected format)
        t0 = time.perf_counter()
        pygame.surfarray.blit_array(self.display_surface, img_np.swapaxes(0, 1))
        timings['blit'] = time.perf_counter() - t0
        
        return self.display_surface, timings
    
    def _resize_image(self, new_width, new_height):
        """Helper to resize image and recreate surfaces (keeps image on device)"""
        # Resize image using bilinear interpolation (stays on device)
        resized = F.interpolate(self.image.unsqueeze(0), 
                               size=(new_height, new_width), 
                               mode='bilinear', align_corners=False)
        self.image = resized.squeeze(0)
        
        # Re-normalize to sphere (interpolation takes us slightly off)
        self.image = F.normalize(self.image, dim=0, eps=1e-6)
        
        # Update dimensions
        self.img_width = new_width
        self.img_height = new_height
        
        # Recreate display surface and perturbation mask
        self.display_surface = pygame.Surface((self.img_width, self.img_height), depth=32)
        self.perturbation_mask = self._create_perturbation_mask()
    
    def toggle_fullscreen(self):
        """Toggle between windowed and fullscreen mode"""
        self.fullscreen = not self.fullscreen
        
        if self.fullscreen:
            # Get screen resolution and set fullscreen
            info = pygame.display.Info()
            self.screen = pygame.display.set_mode((info.current_w, info.current_h), pygame.FULLSCREEN)
            
            # Calculate fullscreen image size (preserve aspect ratio)
            target_size = min(info.current_w, info.current_h)
            self._resize_image(target_size, target_size)
        else:
            # Return to windowed mode
            self.screen = pygame.display.set_mode(self.original_size)
            self._resize_image(self.original_img_size[0], self.original_img_size[1])
    
    def handle_button_click(self, pos):
        """Handle button clicks"""
        for button in self.buttons:
            if button["rect"].collidepoint(pos):
                action = button["action"]
                if action == "randomize_image":
                    self.image = self._random_image()
                elif action == "randomize_kernel":
                    self.conv_processor.randomize_kernels()
                elif action == "change_colors":
                    # Regenerate color mapping basis
                    if hasattr(self.mapper, 'update_basis'):
                        self.mapper.update_basis()
                        print("Color mapping updated")
                return True
        return False
    
    def handle_slider_drag(self, pos, slider):
        """Handle slider dragging"""
        if slider["dragging"]:
            # Update handle position
            slider["handle_x"] = max(slider["rect"].left, 
                                    min(pos[0], slider["rect"].right))
            # Update value with min/max range
            slider_pos = (slider["handle_x"] - slider["rect"].left) / slider["rect"].width
            min_val = slider["min_value"]
            max_val = slider["max_value"]
            value = min_val + slider_pos * (max_val - min_val)
            setattr(slider["target_obj"], slider["value_attr"], value)
    
    def take_screenshot(self):
        """Save current image as PNG with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Remove last 3 digits of microseconds
        filename = f"screenshot_{timestamp}.png"
        filepath = os.path.join(self.screenshot_dir, filename)
        
        # Save the current display surface
        pygame.image.save(self.display_surface, filepath)
        print(f"Screenshot saved: {filepath}")
    
    def draw_fps(self):
        """Draw FPS counter in top-left corner"""
        fps = self.clock.get_fps()
        fps_text = f"FPS: {fps:.1f}"
        if self.paused:
            fps_text += " [PAUSED]"
        fps_surf = self.font.render(fps_text, True, TEXT_COLOR)
        # Add semi-transparent background for readability
        fps_bg = pygame.Surface((fps_surf.get_width() + 10, fps_surf.get_height() + 6))
        fps_bg.set_alpha(180)
        fps_bg.fill(BLACK)
        self.screen.blit(fps_bg, (5, 5))
        self.screen.blit(fps_surf, (10, 8))
    
    def draw_timing(self):
        """Draw timing diagnostics"""
        if not self.show_timing:
            return
        
        # Calculate averages
        timings = {}
        for key, history in self.timing_history.items():
            if history:
                timings[key] = sum(history) / len(history) * 1000  # Convert to ms
            else:
                timings[key] = 0.0
        
        # Create timing display with detailed breakdown
        y_offset = 35
        lines = [
            f"=== Timing (ms/frame) ===",
            f"Drift:  {timings['drift']:.2f}",
            f"Noise:  {timings['noise']:.2f}",
            f"Conv:   {timings['conv']:.2f}",
            f"",
            f"--- Render Breakdown ---",
            f"  Map:     {timings['render_map']:.2f}",
            f"  Sync:    {timings['render_sync']:.2f}",
            f"  CPU:     {timings['render_cpu']:.2f}",
            f"  Numpy:   {timings['render_numpy']:.2f}",
            f"  Blit:    {timings['render_blit']:.2f}",
            f"  UI:      {timings['render_ui']:.2f}",
            f"  Flip:    {timings['render_flip']:.2f}",
            f"  Total:   {timings['render_total']:.2f}",
            f"",
            f"Frame:  {timings['total']:.2f}",
            f"Device: {self.conv_processor.device}"
        ]
        
        # Find max width for background
        max_width = max(self.font.size(line)[0] for line in lines)
        
        # Semi-transparent background
        bg = pygame.Surface((max_width + 20, len(lines) * 25 + 10))
        bg.set_alpha(180)
        bg.fill(BLACK)
        self.screen.blit(bg, (5, y_offset))
        
        # Draw text
        for i, line in enumerate(lines):
            text_surf = self.font.render(line, True, TEXT_COLOR)
            self.screen.blit(text_surf, (10, y_offset + 5 + i * 25))
    
    def draw_ui(self):
        """Draw UI elements"""
        # Draw buttons
        for button in self.buttons:
            color = BUTTON_HOVER if button["hovered"] else BUTTON_COLOR
            pygame.draw.rect(self.screen, color, button["rect"], border_radius=5)
            pygame.draw.rect(self.screen, WHITE, button["rect"], width=2, border_radius=5)
            
            # Draw text
            text_surf = self.font.render(button["text"], True, TEXT_COLOR)
            text_rect = text_surf.get_rect(center=button["rect"].center)
            self.screen.blit(text_surf, text_rect)
        
        # Draw sliders
        for slider in self.sliders:
            # Label
            label_surf = self.font.render(slider["label"], True, TEXT_COLOR)
            self.screen.blit(label_surf, (slider["rect"].x, slider["rect"].y - 25))
            
            # Slider track
            pygame.draw.rect(self.screen, GRAY, slider["rect"], border_radius=3)
            pygame.draw.rect(self.screen, WHITE, slider["rect"], width=2, border_radius=3)
            
            # Slider handle
            handle_radius = 10
            pygame.draw.circle(self.screen, WHITE, 
                             (int(slider["handle_x"]), slider["rect"].centery), 
                             handle_radius)
            
            # Value display
            value = getattr(slider["target_obj"], slider["value_attr"])
            value_text = f"{value:.4f}"
            value_surf = self.font.render(value_text, True, TEXT_COLOR)
            self.screen.blit(value_surf, (slider["rect"].x, slider["rect"].y + 25))
        
        # Instructions
        instructions = [
            "Keys:",
            "I - Randomize Image",
            "K - Randomize Kernels",
            "C - Change Colors",
            "F - Fullscreen",
            "T - Toggle Timing",
            "SPACE - Pause/Play",
            "S - Screenshot",
            "",
            "Click & Hold:",
            "Perturb Image",
            "",
            "ESC - Quit"
        ]
        y_offset = self.window_height - 200
        for i, line in enumerate(instructions):
            text_surf = self.font.render(line, True, TEXT_COLOR)
            self.screen.blit(text_surf, (self.window_width - 200, y_offset + i * 25))
    
    def run(self):
        """Main game loop"""
        while self.running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_i:
                        self.image = self._random_image()
                    elif event.key == pygame.K_k:
                        self.conv_processor.randomize_kernels()
                    elif event.key == pygame.K_c:
                        # Change color mapping
                        if hasattr(self.mapper, 'update_basis'):
                            self.mapper.update_basis()
                            print("Color mapping updated")
                    elif event.key == pygame.K_f:
                        # Toggle fullscreen
                        self.toggle_fullscreen()
                    elif event.key == pygame.K_t:
                        # Toggle timing display
                        self.show_timing = not self.show_timing
                    elif event.key == pygame.K_SPACE:
                        # Toggle pause
                        self.paused = not self.paused
                    elif event.key == pygame.K_s:
                        # Take screenshot
                        self.take_screenshot()
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        # Check sliders first
                        slider_clicked = False
                        for slider in self.sliders:
                            if slider["rect"].collidepoint(event.pos):
                                slider["dragging"] = True
                                self.handle_slider_drag(event.pos, slider)
                                slider_clicked = True
                                break
                        
                        # Check buttons if no slider was clicked
                        if not slider_clicked and not self.handle_button_click(event.pos):
                            # Start mouse perturbation
                            self.mouse_pressed = True
                            self.mouse_press_start = pygame.time.get_ticks()
                            self.mouse_pos = event.pos
                
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        for slider in self.sliders:
                            slider["dragging"] = False
                        self.mouse_pressed = False
                
                elif event.type == pygame.MOUSEMOTION:
                    # Update button hover states
                    for button in self.buttons:
                        button["hovered"] = button["rect"].collidepoint(event.pos)
                    
                    # Handle slider dragging
                    for slider in self.sliders:
                        if slider["dragging"]:
                            self.handle_slider_drag(event.pos, slider)
                    
                    # Update mouse position for perturbation
                    if self.mouse_pressed:
                        self.mouse_pos = event.pos
            
            frame_start = time.perf_counter()
            
            # Only process when not paused
            if not self.paused:
                # Get heart signal and pump (impulsive heartbeat signal)
                heart_signal = self.heart.beat()
                pump_signal = self.heart.get_pump_signal()
                
                # Drift and noise application
                t0 = time.perf_counter()
                # Change drift direction (random walk on sphere with momentum)
                # Use pump signal for more impulsive, visceral heartbeat feeling
                self.conv_processor.change_drift(0.02 * pump_signal)
                
                # Apply drift using heart signal modulated by pump
                # This creates a strong "thump" during each heartbeat
                drift_signal = heart_signal * (1.0 + 3.0 * pump_signal)
                self.conv_processor.apply_drift(drift_signal * self.drift_scale)
                t1 = time.perf_counter()
                self.timing_history['drift'].append(t1 - t0)
                
                # Apply image noise and mouse perturbation
                t0 = time.perf_counter()
                self.apply_image_noise()
                self.apply_mouse_perturbation()
                t1 = time.perf_counter()
                self.timing_history['noise'].append(t1 - t0)
                
                # Apply convolution
                t0 = time.perf_counter()
                self.image = self.conv_processor.apply(self.image, self.residual_alpha)
                t1 = time.perf_counter()
                self.timing_history['conv'].append(t1 - t0)
            
            # Render
            render_start = time.perf_counter()
            
            self.screen.fill(BLACK)
            surface, render_timings = self.image_to_surface(self.image)
            
            # Store detailed render timings
            self.timing_history['render_map'].append(render_timings['map'])
            self.timing_history['render_sync'].append(render_timings['sync'])
            self.timing_history['render_cpu'].append(render_timings['cpu'])
            self.timing_history['render_numpy'].append(render_timings['numpy'])
            self.timing_history['render_blit'].append(render_timings['blit'])
            
            # Calculate display position and draw appropriate UI
            t0 = time.perf_counter()
            if self.fullscreen:
                screen_w, screen_h = self.screen.get_size()
                x = (screen_w - self.img_width) // 2
                y = (screen_h - self.img_height) // 2
                self.display_offset = (x, y)
            else:
                self.display_offset = (IMAGE_MARGIN, IMAGE_MARGIN)
                self.draw_ui()  # Draw full UI only in windowed mode
            
            self.screen.blit(surface, self.display_offset)
            self.draw_fps()  # Always draw FPS counter
            self.draw_timing()  # Draw timing if enabled
            t1 = time.perf_counter()
            self.timing_history['render_ui'].append(t1 - t0)
            
            t0 = time.perf_counter()
            pygame.display.flip()
            t1 = time.perf_counter()
            self.timing_history['render_flip'].append(t1 - t0)
            
            render_end = time.perf_counter()
            self.timing_history['render_total'].append(render_end - render_start)
            
            frame_end = time.perf_counter()
            self.timing_history['total'].append(frame_end - frame_start)
            
            self.clock.tick(FPS)
        
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    # Use default Seraphim (multi-layer) soul with 60 BPM heart rate
    app = ConvolutionArt()
    
    # Or customize the configuration:
    # app = ConvolutionArt(bpm=80)  # Faster heart rate
    # app = ConvolutionArt(conv_processor=Seraphim(kernel_size=5, num_layers=3, sphere_dim=3), bpm=120)
    # app = ConvolutionArt(conv_processor=Seraphim(kernel_size=7, num_layers=5, sphere_dim=5), bpm=60)  # 5D sphere
    
    app.run()
