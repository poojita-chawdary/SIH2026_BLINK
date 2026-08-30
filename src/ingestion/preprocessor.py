"""
Geospatial Preprocessor & Overlapping Tile Splitter/Stitcher.
Handles multi-resolution channel alignment, Hann-window blending for seamless
large-scale full-disk inference, and RGB false-color composite generation.
"""

from typing import List, Tuple, Union
import numpy as np
import torch


class TileProcessor:
    """
    Splits large geospatial tensors into overlapping tiles for GPU memory optimization
    and seamlessly stitches them back using 2D Hanning window spatial weighting.
    """

    def __init__(
        self,
        tile_size: int = 512,
        overlap: int = 64,
        blend_method: str = "hann",
    ):
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - overlap
        self.blend_method = blend_method
        self._weight_mask = self._create_2d_weight_window(tile_size, blend_method)

    def _create_2d_weight_window(self, size: int, method: str) -> np.ndarray:
        """
        Generates 2D spatial weight window (Hann or linear) for overlap blending.
        """
        if method == "hann":
            w_1d = np.hanning(size)
        else:
            w_1d = np.ones(size)
            ramp = np.linspace(0, 1, self.overlap)
            w_1d[: self.overlap] = ramp
            w_1d[-self.overlap :] = ramp[::-1]

        w_2d = np.outer(w_1d, w_1d)
        w_2d = np.maximum(w_2d, 1e-4)  # Prevent zero-division
        return w_2d.astype(np.float32)

    def split_into_tiles(
        self,
        tensor: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[Tuple[int, int, int, int]], Tuple[int, int]]:
        """
        Slices tensor (B, C, H, W) into list of tiles (B, C, tile_size, tile_size).

        Returns:
            tiles: List of sliced tensor tiles.
            coords: List of (y1, y2, x1, x2) bounding boxes.
            orig_shape: (H, W) original spatial dimensions.
        """
        _, _, h, w = tensor.shape
        tiles = []
        coords = []

        # If tensor is smaller than tile size, pad it
        if h <= self.tile_size and w <= self.tile_size:
            pad_h = self.tile_size - h
            pad_w = self.tile_size - w
            padded = torch.nn.functional.pad(
                tensor, (0, pad_w, 0, pad_h), mode="replicate"
            )
            return [padded], [(0, h, 0, w)], (h, w)

        y_steps = list(range(0, max(1, h - self.tile_size + 1), self.stride))
        if y_steps[-1] + self.tile_size < h:
            y_steps.append(h - self.tile_size)

        x_steps = list(range(0, max(1, w - self.tile_size + 1), self.stride))
        if x_steps[-1] + self.tile_size < w:
            x_steps.append(w - self.tile_size)

        for y in y_steps:
            for x in x_steps:
                y2 = min(y + self.tile_size, h)
                x2 = min(x + self.tile_size, w)
                y1 = max(0, y2 - self.tile_size)
                x1 = max(0, x2 - self.tile_size)

                tile = tensor[:, :, y1:y2, x1:x2]
                tiles.append(tile)
                coords.append((y1, y2, x1, x2))

        return tiles, coords, (h, w)

    def stitch_tiles(
        self,
        tiles: List[torch.Tensor],
        coords: List[Tuple[int, int, int, int]],
        orig_shape: Tuple[int, int],
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        """
        Reconstructs full spatial tensor from processed tiles using weighted overlap averaging.
        """
        h, w = orig_shape
        b, c, _, _ = tiles[0].shape

        output = torch.zeros((b, c, h, w), dtype=torch.float32, device=device)
        weight_acc = torch.zeros((1, 1, h, w), dtype=torch.float32, device=device)
        window = torch.from_numpy(self._weight_mask).unsqueeze(0).unsqueeze(0).to(device)

        for tile, (y1, y2, x1, x2) in zip(tiles, coords):
            th, tw = y2 - y1, x2 - x1
            cur_window = window[:, :, :th, :tw]

            output[:, :, y1:y2, x1:x2] += tile.to(device) * cur_window
            weight_acc[:, :, y1:y2, x1:x2] += cur_window

        weight_acc = torch.clamp(weight_acc, min=1e-4)
        output = output / weight_acc
        return output


class GeoNormalizer:
    """
    Multi-spectral radiance normalization and false-color composite synthesis.
    """

    @staticmethod
    def create_false_color_composite(
        vis_channel: np.ndarray,
        wv_channel: np.ndarray,
        tir_channel: np.ndarray,
    ) -> np.ndarray:
        """
        Generates standard INSAT-3DS False Color Composite (RGB: VIS - WV - TIR1).
        Provides high contrast distinguishing land (greenish), ocean (dark),
        active storm cores (bright cyan/white), and upper-troposphere dry air (dark red/blue).

        Returns:
            uint8 RGB image array of shape (H, W, 3) with range [0, 255].
        """
        # Normalize each band to [0, 1]
        r = np.clip((vis_channel - 0.0) / 100.0, 0.0, 1.0)
        # Invert WV: colder/wetter is brighter
        g = np.clip((280.0 - wv_channel) / 90.0, 0.0, 1.0)
        # Invert TIR1: colder convective clouds are brighter
        b = np.clip((330.0 - tir_channel) / 150.0, 0.0, 1.0)

        rgb = np.stack([r, g, b], axis=-1)
        rgb_uint8 = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        return rgb_uint8

    @staticmethod
    def tensor_to_rgb_preview(tensor: torch.Tensor, mode: str = "enhanced_ir") -> np.ndarray:
        """
        Converts a normalized tensor (1, C, H, W) to displayable RGB preview array (H, W, 3) [0..255].
        Produces high-contrast, vivid meteorological satellite imagery with clearly defined clouds,
        deep oceanic basins, and luminous convective cores.
        """
        arr = tensor.detach().cpu().squeeze(0).numpy()
        c, h, w = arr.shape

        if c >= 3:
            vis = np.clip(arr[0], 0.0, 1.0)
            wv = np.clip(arr[1], 0.0, 1.0)
            tir = np.clip(arr[2], 0.0, 1.0)

            # Cloud intensity is strongest where TIR is cold (high normalized value) or VIS is high
            cloud_intensity = np.maximum(tir * 1.15, vis * 0.95)
            cloud_norm = np.clip(cloud_intensity, 0.0, 1.0)
            cloud_gamma = np.power(cloud_norm, 0.8)

            # Build rich false-color meteorological composite
            # Deep ocean/land background: (8, 18, 38)
            bg_r = 10.0
            bg_g = 18.0
            bg_b = 36.0

            # Mid-level clouds: soft cyan/white
            # Severe convective cloud tops: brilliant luminous white with cyan tint
            r_out = bg_r * (1.0 - cloud_gamma) + (cloud_gamma * 255.0)
            g_out = bg_g * (1.0 - cloud_gamma) + (cloud_gamma * 252.0)
            b_out = bg_b * (1.0 - cloud_gamma) + (cloud_gamma * 248.0)

            # Cold convective cores (TIR > 0.82) glow with intense bright white/cyan
            severe_mask = np.clip((tir - 0.80) / 0.18, 0.0, 1.0)
            r_out = np.clip(r_out * (1.0 - severe_mask * 0.15) + severe_mask * 255.0, 0.0, 255.0)
            g_out = np.clip(g_out * (1.0 - severe_mask * 0.05) + severe_mask * 255.0, 0.0, 255.0)
            b_out = np.clip(b_out + severe_mask * 20.0, 0.0, 255.0)

            rgb = np.stack([r_out, g_out, b_out], axis=-1)
            return rgb.astype(np.uint8)
        else:
            gray = np.clip(arr[0] * 255.0, 0, 255).astype(np.uint8)
            return np.stack([gray, gray, gray], axis=-1)
