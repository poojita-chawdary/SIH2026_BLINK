"""
Unit tests for MOSDAC ingestion, calibration, and synthetic generator.
"""

from pathlib import Path
import numpy as np
import pytest
import torch

from src.ingestion.mosdac_parser import MOSDACParser, SyntheticMOSDACSimulator
from src.ingestion.preprocessor import GeoNormalizer, TileProcessor


def test_synthetic_data_generation_and_hdf5_io(tmp_path: Path):
    # 1. Generate synthetic cyclone frame
    data = SyntheticMOSDACSimulator.generate_cyclone_frame(grid_size=(128, 128), t_normalized=0.5)

    assert "IMG_VIS" in data
    assert "IMG_WV" in data
    assert "IMG_TIR1" in data
    assert data["IMG_VIS"].shape == (128, 128)
    assert data["IMG_TIR1"].min() >= 180.0
    assert data["IMG_TIR1"].max() <= 330.0

    # 2. Save to HDF5
    h5_file = tmp_path / "test_insat3ds.h5"
    SyntheticMOSDACSimulator.save_to_hdf5(h5_file, data)
    assert h5_file.exists()

    # 3. Read via MOSDACParser
    parser = MOSDACParser(channels=["IMG_VIS", "IMG_WV", "IMG_TIR1"])
    loaded = parser.read_hdf5(h5_file, target_size=(128, 128))
    assert "IMG_VIS" in loaded
    np.testing.assert_allclose(loaded["IMG_VIS"], data["IMG_VIS"], rtol=1e-4)

    # 4. Normalize to tensor
    tensor = parser.to_normalized_tensor(loaded)
    assert tensor.shape == (1, 3, 128, 128)
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0


def test_tile_processor_split_and_stitch():
    processor = TileProcessor(tile_size=64, overlap=16, blend_method="hann")
    dummy_input = torch.rand(1, 3, 128, 128)

    tiles, coords, orig_shape = processor.split_into_tiles(dummy_input)
    assert len(tiles) > 1
    assert orig_shape == (128, 128)

    # Reconstruct
    reconstructed = processor.stitch_tiles(tiles, coords, orig_shape)
    assert reconstructed.shape == dummy_input.shape
    # Reconstructed should closely match original
    torch.testing.assert_close(reconstructed, dummy_input, rtol=1e-3, atol=1e-3)


def test_false_color_composite():
    vis = np.random.uniform(0, 100, (64, 64)).astype(np.float32)
    wv = np.random.uniform(190, 280, (64, 64)).astype(np.float32)
    tir1 = np.random.uniform(180, 330, (64, 64)).astype(np.float32)

    rgb = GeoNormalizer.create_false_color_composite(vis, wv, tir1)
    assert rgb.shape == (64, 64, 3)
    assert rgb.dtype == np.uint8
    assert rgb.min() >= 0
    assert rgb.max() <= 255
