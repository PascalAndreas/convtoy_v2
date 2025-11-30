"""
ECG signal simulator for driving convolution drift.

This module provides a realistic ECG (electrocardiogram) signal generator
that can be used to modulate visual parameters based on heart rate.
"""

import numpy as np


class Heart:
    """
    Simulates a realistic ECG signal with P, QRS, and T waves.
    
    The ECG waveform is constructed using Gaussian-like functions to model
    the different components of a heartbeat. The signal can be sampled at
    any rate and with any heart rate (BPM).
    
    Includes natural heart rate variability (HRV) and respiratory sinus arrhythmia
    for realistic beat-to-beat variations.
    
    Features:
    - Beat timing variability (HRV random walk)
    - Respiratory sinus arrhythmia (breathing-heart coupling)
    - Amplitude variations per beat
    - Wave morphology variations (each P, Q, R, S, T wave varies slightly in
      timing, width, and amplitude from beat to beat)
    """
    
    def __init__(self, bpm=60, sample_rate=60, amplitude=1.0, 
                 hrv_amount=0.05, breathing_rate=0.25):
        """
        Initialize the ECG simulator.
        
        Args:
            bpm: Beats per minute (default: 60)
            sample_rate: Number of samples per second (default: 60, matching typical FPS)
            amplitude: Maximum amplitude of the signal (default: 1.0)
            hrv_amount: Heart rate variability as fraction of beat period (default: 0.05 = 5%)
            breathing_rate: Breaths per second for respiratory sinus arrhythmia (default: 0.25 = 15 breaths/min)
        """
        self.base_bpm = bpm
        self.sample_rate = sample_rate
        self.base_amplitude = amplitude
        self.hrv_amount = hrv_amount
        self.breathing_rate = breathing_rate
        
        # Calculate base beat period in seconds
        self.base_beat_period = 60.0 / bpm
        
        # Current beat period (will vary)
        self.current_beat_period = self.base_beat_period
        
        # Internal state: current time in seconds
        self.time = 0.0
        
        # Time since last beat (for detecting beat transitions)
        self.time_since_beat = 0.0
        
        # Time step per sample
        self.dt = 1.0 / sample_rate
        
        # HRV state: random walk for beat-to-beat variation
        self.hrv_offset = 0.0
        
        # Amplitude variation
        self.current_amplitude = amplitude
        
        # Base ECG wave parameters (as fractions of beat period)
        # These define the baseline timing and shape of P, Q, R, S, T waves
        # These remain constant as reference points
        self.base_params = {
            # P wave (atrial depolarization)
            'p_center': 0.15,    # 15% into the cycle
            'p_width': 0.08,     # Width of P wave
            'p_amplitude': 0.15, # 15% of max amplitude
            
            # Q wave (start of ventricular depolarization)
            'q_center': 0.30,
            'q_width': 0.03,
            'q_amplitude': -0.15, # Negative (downward)
            
            # R wave (main spike of ventricular depolarization)
            'r_center': 0.35,
            'r_width': 0.04,
            'r_amplitude': 1.0,  # Full amplitude
            
            # S wave (end of ventricular depolarization)
            's_center': 0.40,
            's_width': 0.06,     # Wider for more prominent negative
            's_amplitude': -0.35, # More negative (downward)
            
            # T wave (ventricular repolarization)
            't_center': 0.65,
            't_width': 0.12,
            't_amplitude': 0.20,  # Slightly reduced
            
            # ST segment baseline shift (between S and T)
            'st_center': 0.50,
            'st_width': 0.15,
            'st_amplitude': -0.05, # Slight negative baseline during ST segment
        }
        
        # Current wave parameters (will vary each beat)
        # Start as a copy of base_params, will be updated by _update_variability()
        self.wave_params = self.base_params.copy()
        
        # Initialize with first variation
        self._update_variability()
    
    def _gaussian_wave(self, t, center, width, amplitude):
        """
        Generate a Gaussian-like wave component.
        
        Args:
            t: Time within beat cycle (0 to 1)
            center: Center position of the wave (0 to 1)
            width: Width of the wave (standard deviation)
            amplitude: Amplitude of the wave
            
        Returns:
            Wave value at time t
        """
        return amplitude * np.exp(-((t - center) ** 2) / (2 * width ** 2))
    
    def _generate_single_beat(self, phase):
        """
        Generate ECG signal value for a given phase within a beat.
        
        Args:
            phase: Phase within beat cycle (0 to 1)
            
        Returns:
            ECG signal value
        """
        signal = 0.0
        
        # Add all wave components
        signal += self._gaussian_wave(phase, 
                                     self.wave_params['p_center'],
                                     self.wave_params['p_width'],
                                     self.wave_params['p_amplitude'])
        
        signal += self._gaussian_wave(phase,
                                     self.wave_params['q_center'],
                                     self.wave_params['q_width'],
                                     self.wave_params['q_amplitude'])
        
        signal += self._gaussian_wave(phase,
                                     self.wave_params['r_center'],
                                     self.wave_params['r_width'],
                                     self.wave_params['r_amplitude'])
        
        signal += self._gaussian_wave(phase,
                                     self.wave_params['s_center'],
                                     self.wave_params['s_width'],
                                     self.wave_params['s_amplitude'])
        
        signal += self._gaussian_wave(phase,
                                     self.wave_params['t_center'],
                                     self.wave_params['t_width'],
                                     self.wave_params['t_amplitude'])
        
        # Add ST segment baseline shift
        signal += self._gaussian_wave(phase,
                                     self.wave_params['st_center'],
                                     self.wave_params['st_width'],
                                     self.wave_params['st_amplitude'])
        
        return signal * self.current_amplitude
    
    def _update_variability(self):
        """
        Update heart rate variability for the next beat.
        
        Combines:
        1. Random walk (beat-to-beat HRV)
        2. Respiratory sinus arrhythmia (breathing cycle)
        3. Slight amplitude variations
        4. Wave shape and timing variations
        """
        # Random walk for HRV (small random changes each beat)
        hrv_change = np.random.randn() * 0.3 * self.hrv_amount
        self.hrv_offset = np.clip(self.hrv_offset + hrv_change, 
                                  -self.hrv_amount * 2, 
                                  self.hrv_amount * 2)
        
        # Respiratory sinus arrhythmia (breathing affects heart rate)
        # Heart rate increases during inspiration, decreases during expiration
        breathing_phase = 2 * np.pi * self.breathing_rate * self.time
        respiratory_effect = np.sin(breathing_phase) * self.hrv_amount * 0.8
        
        # Combine effects to get current beat period
        total_variation = self.hrv_offset + respiratory_effect
        self.current_beat_period = self.base_beat_period * (1.0 + total_variation)
        
        # Small amplitude variations (±3%)
        amplitude_variation = 1.0 + np.random.randn() * 0.03
        self.current_amplitude = self.base_amplitude * amplitude_variation
        
        # Update wave parameters with small variations from base
        # Each wave component gets slightly different timing/shape each beat
        self.wave_params = {}
        for key, base_value in self.base_params.items():
            if '_center' in key:
                # Timing variation: ±2% of beat cycle
                variation = np.random.randn() * 0.02
                self.wave_params[key] = base_value + variation
            elif '_width' in key:
                # Width variation: ±10% of base width
                variation = 1.0 + np.random.randn() * 0.1
                self.wave_params[key] = base_value * variation
            elif '_amplitude' in key:
                # Amplitude variation: ±8% of base amplitude
                variation = 1.0 + np.random.randn() * 0.08
                self.wave_params[key] = base_value * variation
            else:
                # Unknown parameter, keep as is
                self.wave_params[key] = base_value
    
    def beat(self):
        """
        Get the next sample of the ECG signal.
        
        This should be called once per frame to get the current ECG value.
        The signal will automatically loop through heartbeat cycles with
        natural variability.
        
        Returns:
            Current ECG signal value (normalized around 0, typically -amplitude to +amplitude)
        """
        # Check if we're starting a new beat
        if self.time_since_beat >= self.current_beat_period:
            # New beat! Update variability for next beat
            self._update_variability()
            self.time_since_beat = 0.0
        
        # Calculate phase within current beat (0 to 1)
        phase = self.time_since_beat / self.current_beat_period
        
        # Generate signal value for this phase
        signal = self._generate_single_beat(phase)
        
        # Advance time
        self.time += self.dt
        self.time_since_beat += self.dt
        
        return signal
    
    def set_bpm(self, bpm):
        """
        Change the heart rate.
        
        Args:
            bpm: New beats per minute
        """
        self.base_bpm = max(30, min(bpm, 200))  # Clamp to reasonable range
        self.base_beat_period = 60.0 / self.base_bpm
        self.current_beat_period = self.base_beat_period
    
    def reset(self):
        """Reset the simulator to the beginning of a beat cycle."""
        self.time = 0.0
        self.time_since_beat = 0.0
        self.hrv_offset = 0.0
    
    def get_phase(self):
        """
        Get current phase within the beat cycle.
        
        Returns:
            Phase from 0.0 (start of beat) to 1.0 (end of beat)
        """
        return self.time_since_beat / self.current_beat_period
    
    def set_hrv_amount(self, hrv_amount):
        """
        Change the amount of heart rate variability.
        
        Args:
            hrv_amount: HRV as fraction of beat period (e.g., 0.05 = 5%)
        """
        self.hrv_amount = max(0.0, min(hrv_amount, 0.2))  # Clamp to 0-20%
    
    def set_breathing_rate(self, breaths_per_minute):
        """
        Change the breathing rate for respiratory sinus arrhythmia.
        
        Args:
            breaths_per_minute: Breaths per minute (typical: 12-20)
        """
        self.breathing_rate = breaths_per_minute / 60.0  # Convert to breaths per second
    
    def get_pump_signal(self, width=0.08):
        """
        Get an impulsive "pump" signal that spikes during R wave.
        
        This provides a short, intense burst during each heartbeat that makes
        the heart's rhythm much more viscerally apparent in visuals.
        
        Args:
            width: Width of the pump impulse as fraction of beat cycle (default: 0.08)
            
        Returns:
            Pump signal value (0 to 1, spikes near R wave)
        """
        phase = self.get_phase()
        r_center = self.wave_params['r_center']
        
        # Create narrow Gaussian pulse around R wave
        pump = np.exp(-((phase - r_center) ** 2) / (2 * (width ** 2)))
        
        return pump * self.current_amplitude

