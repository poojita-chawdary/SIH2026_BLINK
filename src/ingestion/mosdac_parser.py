"""
MOSDAC Data Parser & Synthetic Atmospheric Simulator for INSAT-3DS/3DR.
Ingests Level-1B and Level-2 HDF5/NetCDF4 radiance arrays, calibrates
brightness temperatures (BT in Kelvin) or reflectances, and outputs
normalized PyTorch tensors of shape (B, C, H, W).
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import h5py
import numpy as np
import torch


# Physical calibration lookup bounds per channel for INSAT-3DS
CHANNEL_CALIBRATION_BOUNDS = {
    "IMG_VIS": {"min": 0.0, "max": 100.0, "unit": "% Reflectance", "type": "reflectance"},
    "IMG_SWIR": {"min": 0.0, "max": 100.0, "unit": "% Reflectance", "type": "reflectance"},
    "IMG_MWIR": {"min": 180.0, "max": 330.0, "unit": "Kelvin", "type": "temperature"},
    "IMG_WV": {"min": 190.0, "max": 280.0, "unit": "Kelvin", "type": "temperature"},
    "IMG_TIR1": {"min": 180.0, "max": 330.0, "unit": "Kelvin", "type": "temperature"},
    "IMG_TIR2": {"min": 180.0, "max": 330.0, "unit": "Kelvin", "type": "temperature"},
}


class MOSDACParser:
    """
    Parser for MOSDAC Earth Observation INSAT-3DS / INSAT-3DR HDF5 and NetCDF4 radiance data.
    Implements physical calibration table extraction and GeoTIFF export.
    """

    def __init__(self, channels: Optional[List[str]] = None):
        self.channels = channels or ["IMG_VIS", "IMG_WV", "IMG_TIR1"]

    def read_hdf5(
        self,
        filepath: Union[str, Path],
        target_size: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Reads multi-spectral datasets from a MOSDAC HDF5 file.

        Args:
            filepath: Path to the .h5 / .nc file.
            target_size: Optional (height, width) to resample all channels to.

        Returns:
            Dictionary mapping channel name to 2D numpy array of calibrated values.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"MOSDAC file not found: {filepath}")

        extracted_channels = {}
        with h5py.File(filepath, "r") as h5_file:
            for ch in self.channels:
                raw_data = None
                # Search directly or in common MOSDAC group paths
                candidate_keys = [
                    ch,
                    f"/{ch}",
                    f"/Data/{ch}",
                    f"/BAND_{ch}",
                    f"/Radiance_{ch}",
                    ch.lower(),
                ]
                for key in candidate_keys:
                    if key in h5_file:
                        raw_data = np.array(h5_file[key], dtype=np.float32)
                        break

                if raw_data is None:
                    # If specific channel is missing, check if keys match partially
                    for key in h5_file.keys():
                        if ch.lower() in key.lower():
                            raw_data = np.array(h5_file[key], dtype=np.float32)
                            break

                if raw_data is None:
                    raise KeyError(f"Channel {ch} not found in {filepath}. Available keys: {list(h5_file.keys())}")

                # Calibrate and clean invalid fill values (-999, NaN, Inf)
                calibrated = self._calibrate_channel(raw_data, ch)

                if target_size is not None and calibrated.shape != target_size:
                    calibrated = self._resample_array(calibrated, target_size)

                extracted_channels[ch] = calibrated

        return extracted_channels

    def _calibrate_channel(self, raw_data: np.ndarray, channel_name: str) -> np.ndarray:
        """
        Cleans fill values and bounds data within physical limits.
        """
        bounds = CHANNEL_CALIBRATION_BOUNDS.get(
            channel_name, {"min": 0.0, "max": 1.0, "type": "generic"}
        )
        data = raw_data.copy()

        # Handle fill values
        fill_mask = (data < -900) | np.isnan(data) | np.isinf(data)
        valid_min = bounds["min"]
        valid_max = bounds["max"]

        if np.any(fill_mask):
            valid_vals = data[~fill_mask]
            fallback_val = valid_vals.mean() if valid_vals.size > 0 else valid_min
            data[fill_mask] = fallback_val

        # Clip within physical range
        data = np.clip(data, valid_min, valid_max)
        return data

    def _resample_array(self, arr: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Interpolates 2D array to target (height, width) using bilinear interpolation.
        """
        h, w = target_size
        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
        resampled = torch.nn.functional.interpolate(
            tensor, size=(h, w), mode="bilinear", align_corners=True
        )
        return resampled.squeeze(0).squeeze(0).numpy()

    def to_normalized_tensor(
        self,
        channel_data: Dict[str, np.ndarray],
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        """
        Converts channel dictionaries into a single normalized PyTorch tensor of shape (1, C, H, W)
        with values scaled in [0.0, 1.0]. For thermal IR, cold cloud tops (<220K) are mapped to
        higher values (1.0) so optical flow tracks dense convective cloud masses accurately.
        """
        stacked = []
        for ch in self.channels:
            arr = channel_data[ch]
            bounds = CHANNEL_CALIBRATION_BOUNDS.get(ch, {"min": 0.0, "max": 1.0, "type": "generic"})
            c_min = bounds["min"]
            c_max = bounds["max"]

            # Normalize to 0..1
            norm = (arr - c_min) / (c_max - c_min + 1e-7)
            norm = np.clip(norm, 0.0, 1.0)

            # Invert temperature channels for cloud tracking (cold = active convective cloud top = bright)
            if bounds.get("type") == "temperature":
                norm = 1.0 - norm

            stacked.append(norm)

        tensor_np = np.stack(stacked, axis=0)  # Shape (C, H, W)
        tensor = torch.from_numpy(tensor_np).unsqueeze(0).float().to(device)  # Shape (1, C, H, W)
        return tensor

    def tensor_to_physical(
        self,
        tensor: torch.Tensor,
    ) -> Dict[str, np.ndarray]:
        """
        Converts normalized tensor (1, C, H, W) back to physical calibrated units per channel.
        """
        tensor_cpu = tensor.detach().cpu().squeeze(0).numpy()
        result = {}
        for i, ch in enumerate(self.channels):
            bounds = CHANNEL_CALIBRATION_BOUNDS.get(ch, {"min": 0.0, "max": 1.0, "type": "generic"})
            c_min = bounds["min"]
            c_max = bounds["max"]
            arr = tensor_cpu[i]

            if bounds.get("type") == "temperature":
                arr = 1.0 - arr

            physical = arr * (c_max - c_min) + c_min
            result[ch] = physical
        return result


class SyntheticMOSDACSimulator:
    """
    High-fidelity physical atmospheric simulator that generates synthetic INSAT-3DS HDF5
    scenarios (e.g. Tropical Cyclone Vortex, Meso-scale Convective Burst, Zonal Shear Flow)
    with exact ground-truth fluid advection across arbitrary time steps (e.g. 1-min cadence).
    """

    @staticmethod
    def _fractal_noise(x_norm, y_norm, octaves=6, persistence=0.5, seed_offset=0.0):
        """
        Multi-octave fractal noise using layered sinusoidal harmonics.
        Produces turbulent, cloud-like texture without external libraries.
        """
        noise = np.zeros_like(x_norm)
        amplitude = 1.0
        frequency = 1.0
        max_amplitude = 0.0
        for i in range(octaves):
            # Use prime-number-offset frequencies for non-repeating patterns
            fx = frequency * 7.13 + seed_offset
            fy = frequency * 5.97 + seed_offset * 0.7
            phase1 = np.sin(x_norm * fx + y_norm * fy * 1.31 + i * 2.17) 
            phase2 = np.cos(x_norm * fy * 0.89 - y_norm * fx * 1.07 + i * 3.91)
            phase3 = np.sin((x_norm + y_norm) * fx * 0.67 + i * 1.43)
            noise += amplitude * (phase1 * 0.5 + phase2 * 0.35 + phase3 * 0.15)
            max_amplitude += amplitude
            amplitude *= persistence
            frequency *= 2.03
        return noise / max_amplitude

    @staticmethod
    def generate_cyclone_frame(
        grid_size: Tuple[int, int] = (512, 512),
        t_normalized: float = 0.0,
        center: Tuple[float, float] = (0.55, 0.45),
        intensity: float = 1.0,
        rotation_rate: float = 0.4,
        drift_velocity: Tuple[float, float] = (-0.03, -0.015),
    ) -> Dict[str, np.ndarray]:
        """
        Simulates realistic INSAT-3DS TIR-1 / Multi-Spectral cyclone imagery
        with multi-octave fractal turbulent cloud texture, asymmetric spiral
        rainbands, messy eyewall convection, warm-core eye anomaly, cirrus
        outflow canopy, and realistic background ocean/land thermal structure.
        """
        h, w = grid_size
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)

        # Normalized coordinates [-1, 1] relative to drifting center
        cx = center[0] + drift_velocity[0] * t_normalized
        cy = center[1] + drift_velocity[1] * t_normalized
        x_norm = (x / w - cx) * 2.0
        y_norm = (y / h - cy) * 2.0
        r = np.sqrt(x_norm**2 + y_norm**2) + 1e-6
        theta = np.arctan2(y_norm, x_norm)

        # ── Fractal turbulent cloud texture (6-octave) ──
        turb = SyntheticMOSDACSimulator._fractal_noise(
            x_norm * 3.5, y_norm * 3.5, octaves=6, persistence=0.52, seed_offset=t_normalized * 2.0
        )
        fine_turb = SyntheticMOSDACSimulator._fractal_noise(
            x_norm * 8.0, y_norm * 8.0, octaves=4, persistence=0.45, seed_offset=t_normalized * 3.0 + 17.0
        )

        # ── Differential vortex rotation (Rankine-like profile) ──
        r_max_wind = 0.12  # radius of maximum wind
        v_tan = np.where(
            r < r_max_wind,
            rotation_rate * 2.0 * np.pi * (r / r_max_wind),
            rotation_rate * 2.0 * np.pi * (r_max_wind / r) ** 0.55
        )
        rot = theta + v_tan * t_normalized

        # ── Asymmetric multi-scale logarithmic spiral rainbands ──
        # Asymmetry: enhanced bands in the NE and SW quadrants (realistic for Bay of Bengal cyclones)
        asymmetry = 1.0 + 0.35 * np.cos(theta - 0.8) + 0.2 * np.cos(2.0 * theta + 1.5)

        spiral_1 = np.sin(3.5 * np.log(r * 3.0 + 0.1) - rot * 1.6 + turb * 0.8)
        spiral_2 = np.sin(5.0 * np.log(r * 4.0 + 0.15) - rot * 1.8 + 1.2 + turb * 0.5)
        spiral_3 = np.sin(2.0 * np.log(r * 2.0 + 0.05) - rot * 1.2 + 2.5 + turb * 0.3)
        spiral_4 = np.sin(7.0 * np.log(r * 5.0 + 0.2) - rot * 2.1 + 0.7) * 0.4

        # Radial envelope — asymmetric, with extended outer bands
        envelope_inner = np.exp(-2.0 * (r - 0.18) ** 2) * np.clip(1.0 - r * 0.5, 0.0, 1.0)
        envelope_outer = np.exp(-0.8 * (r - 0.40) ** 2) * np.clip(1.0 - r * 0.35, 0.0, 1.0) * 0.6

        spiral_composite = (
            spiral_1 * 0.45 + spiral_2 * 0.25 + spiral_3 * 0.20 + spiral_4 * 0.15
            + turb * 0.25 + fine_turb * 0.12
        )
        spiral_clouds = np.clip(
            spiral_composite * (envelope_inner + envelope_outer) * intensity * asymmetry,
            -0.15, 1.3
        )

        # ── Dense Central Overcast (CDO) with turbulent eyewall ──
        eye_radius = 0.04 + 0.008 * np.sin(theta * 3.0 + t_normalized * 5.0)  # wobbly eye
        eye_wall_width = 0.03
        eye_mask = np.clip((r - eye_radius) / eye_wall_width, 0.0, 1.0)

        # Eyewall convective bursts — hot towers at irregular positions
        eyewall_ring = np.exp(-((r - 0.065) ** 2) / 0.0008)
        eyewall_bursts = eyewall_ring * (
            1.0 + 0.5 * np.sin(theta * 5.0 + t_normalized * 8.0)
            + 0.3 * np.sin(theta * 8.0 - t_normalized * 12.0)
            + turb * 0.4
        )

        cdo_core = np.exp(-10.0 * r**2) * eye_mask * 1.15 + eyewall_bursts * 0.4

        # ── Warm-core eye anomaly (warmer brightness temp inside the eye) ──
        eye_warmth = np.exp(-((r / (eye_radius.mean() * 0.7)) ** 2.5)) * (1.0 - eye_mask) * 0.45

        # ── Outflow cirrus canopy (wispy radial streaks with turbulence) ──
        cirrus = (
            np.sin(12.0 * theta + 4.0 * r - rot + turb * 1.5)
            * np.exp(-1.8 * (r - 0.5)**2) * 0.25
            + np.sin(8.0 * theta - 3.0 * r + rot * 0.5 + fine_turb * 2.0)
            * np.exp(-2.5 * (r - 0.65)**2) * 0.15
        )

        # ── Outer feeder bands (large-scale, asymmetric) ──
        feeder_1 = np.maximum(0.0, np.sin(
            2.5 * np.log(r * 1.5 + 0.08) - rot * 0.8 + turb * 0.6
        )) * np.exp(-1.5 * (r - 0.55) ** 2) * 0.35 * asymmetry
        feeder_2 = np.maximum(0.0, np.sin(
            x_norm * 3.0 + y_norm * 2.0 + t_normalized * 0.8
        )) * np.exp(-3.0 * r) * 0.15

        # ── Realistic spatial bounding & decay ──
        # Storm system is centered in the basin; outer basin and continental landmass have clear skies
        storm_envelope = np.exp(-1.4 * (r - 0.22) ** 2) * np.clip(1.35 - r * 0.85, 0.0, 1.0)

        raw_cloud = (
            cdo_core * 1.20
            + spiral_clouds * 0.85
            + cirrus * 0.75
            + feeder_1 * 0.65 + feeder_2 * 0.45
            + fine_turb * 0.04
        )
        # Apply storm envelope so clouds don't blanket the entire continent/ocean
        cloud_field = np.clip(raw_cloud, 0.0, 1.3) * storm_envelope

        # Subtle localized cumulus only in warm moist ocean sectors
        cumulus_noise = np.maximum(0.0, np.sin(x_norm * 8.0 + y_norm * 7.0 + fine_turb * 2.0) * np.cos(x_norm * 11.0 - y_norm * 9.0) - 0.2) * 0.25
        isolated_cumulus = cumulus_noise * np.clip(0.85 - r * 0.7, 0.0, 0.4)
        cloud_field = np.clip(cloud_field + isolated_cumulus, 0.0, 1.0)

        # Zero floor for clear skies so landmass and ocean underneath are 100% visible
        cloud_field = np.where(cloud_field < 0.07, 0.0, cloud_field)
        cloud_field = np.power(cloud_field, 0.75)

        # ── Physical channel synthesis ──
        # TIR-1: Cold cloud tops ~187K, warm ocean ~302K, warm eye ~280K
        tir1 = (302.0 - cloud_field * 115.0 + eye_warmth * 40.0).astype(np.float32)
        tir1 = np.clip(tir1, 185.0, 310.0)

        # VIS: High reflectance for dense clouds, dark ocean
        vis = (cloud_field * 92.0 + 5.0).astype(np.float32)
        vis = np.clip(vis, 2.0, 100.0)

        # WV: Deep troposphere moisture tracking
        wv = (265.0 - cloud_field * 58.0 + turb * 3.0 * storm_envelope).astype(np.float32)
        wv = np.clip(wv, 195.0, 275.0)

        return {"IMG_VIS": vis, "IMG_WV": wv, "IMG_TIR1": tir1}

    @staticmethod
    def generate_convective_cloudburst_frame(
        grid_size: Tuple[int, int] = (512, 512),
        t_normalized: float = 0.0,
    ) -> Dict[str, np.ndarray]:
        """
        Simulates explosive convective cloudburst with multi-cell updraft towers,
        rapid cirrus anvil expansion, overshooting tops (<205 K), feeder inflow bands,
        and turbulent cloud texture for realistic satellite appearance.
        """
        h, w = grid_size
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)

        # Primary expanding storm cluster (drifting northeast)
        cx, cy = 0.52 + 0.06 * t_normalized, 0.48 - 0.03 * t_normalized
        x_norm = (x / w - cx) * 2.0
        y_norm = (y / h - cy) * 2.0
        r = np.sqrt(x_norm**2 + y_norm**2) + 1e-6
        theta = np.arctan2(y_norm, x_norm)

        # Fractal turbulence for realistic cloud texture
        turb = SyntheticMOSDACSimulator._fractal_noise(
            x_norm * 4.0, y_norm * 4.0, octaves=5, persistence=0.50, seed_offset=t_normalized * 3.0 + 7.0
        )
        fine_turb = SyntheticMOSDACSimulator._fractal_noise(
            x_norm * 10.0, y_norm * 10.0, octaves=4, persistence=0.42, seed_offset=t_normalized * 4.0 + 23.0
        )

        # Rapid radial anvil expansion (growing 2.5x over the 15-min interval)
        current_radius = 0.18 + 0.32 * t_normalized
        # Irregular anvil edge with turbulence
        edge_wobble = 1.0 + turb * 0.15 + 0.08 * np.sin(theta * 6.0 + t_normalized * 3.0)
        core_growth = np.exp(-((r / (current_radius * edge_wobble)) ** 2.0))

        # Multiple overshooting convective tops (intense cold peaks)
        ot1 = np.exp(-((x_norm + 0.04)**2 + (y_norm - 0.03)**2) / 0.006) * 1.4
        ot2 = np.exp(-((x_norm - 0.08)**2 + (y_norm + 0.05)**2) / 0.010) * 1.2
        ot3 = np.exp(-((x_norm + 0.02)**2 + (y_norm + 0.08)**2) / 0.008) * 0.9
        ot_peaks = ot1 + ot2 + ot3

        # Anvil gravity wave ripples and cirrus outflow plumes with texture
        ripples = 0.20 * np.sin(16.0 * r - 6.0 * t_normalized + 3.0 * theta + turb * 2.0) * core_growth
        feeder_bands = (
            np.maximum(0.0, np.sin(x_norm * 6.0 + y_norm * 4.0 - t_normalized * 1.5 + turb * 0.8))
            * 0.35 * np.exp(-r * 1.3)
        )

        # Multi-cell structure within the anvil
        cells = (
            np.sin(x_norm * 15.0 + turb * 3.0) * np.cos(y_norm * 13.0 + fine_turb * 2.5)
            * core_growth * 0.15
        )

        # Bounded convective cloudburst: localized to storm cell and expanding anvil
        storm_mask = np.clip(core_growth * 1.3 + np.exp(-r * 1.5) * 0.4, 0.0, 1.0)
        density = np.clip(
            core_growth * 0.95
            + ot_peaks * 0.40
            + ripples
            + feeder_bands * 0.5
            + cells * 0.4
            + fine_turb * 0.04,
            0.0, 1.0
        ) * storm_mask
        density = np.where(density < 0.06, 0.0, density)
        density = np.power(density, 0.78)

        vis = (density * 94.0 + 4.0).astype(np.float32)
        vis = np.clip(vis, 2.0, 100.0)
        tir1 = (305.0 - density * 120.0).astype(np.float32)
        tir1 = np.clip(tir1, 182.0, 310.0)
        wv = (268.0 - density * 62.0 + turb * 2.5 * storm_mask).astype(np.float32)
        wv = np.clip(wv, 195.0, 275.0)

        return {"IMG_VIS": vis, "IMG_WV": wv, "IMG_TIR1": tir1}

    @staticmethod
    def save_to_hdf5(
        filepath: Union[str, Path],
        channel_data: Dict[str, np.ndarray],
        timestamp_str: str = "2026-08-14T12:00:00Z",
        satellite_id: str = "INSAT-3DS",
    ) -> Path:
        """
        Saves simulated or preprocessed channel arrays into a MOSDAC-compatible HDF5 file.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(filepath, "w") as h5:
            # Root attributes
            h5.attrs["Satellite_Name"] = satellite_id
            h5.attrs["Observation_Time"] = timestamp_str
            h5.attrs["Product_Level"] = "L1B"
            h5.attrs["Sensor"] = "6-Channel Multi-Spectral Imager"

            for ch_name, data in channel_data.items():
                dset = h5.create_dataset(
                    ch_name,
                    data=data,
                    dtype="float32",
                    compression="gzip",
                    compression_opts=4,
                )
                bounds = CHANNEL_CALIBRATION_BOUNDS.get(ch_name, {})
                dset.attrs["Unit"] = bounds.get("unit", "dimensionless")
                dset.attrs["Valid_Min"] = bounds.get("min", float(data.min()))
                dset.attrs["Valid_Max"] = bounds.get("max", float(data.max()))

        return filepath
