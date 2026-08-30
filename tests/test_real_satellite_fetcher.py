"""
Tests for RealSatelliteFetcher.
"""

from pathlib import Path
import numpy as np
import pytest

from src.ingestion.real_satellite_fetcher import RealSatelliteFetcher


def test_latlon_to_tile():
    # India center approx 18°N, 85°E
    row, col = RealSatelliteFetcher.latlon_to_tile(18.0, 85.0, zoom=6)
    assert 0 <= row < 64
    assert 0 <= col < 128


def test_real_satellite_fetcher_pair(tmp_path):
    fetcher = RealSatelliteFetcher(cache_dir=tmp_path)
    d0, d1, meta = fetcher.fetch_frame_pair(
        date_str="2026-08-28",
        t0_time="10:00",
        region_key="indian_subcontinent",
        target_size=(64, 64),
    )

    assert "IMG_VIS" in d0
    assert "IMG_WV" in d0
    assert "IMG_TIR1" in d0
    assert d0["IMG_TIR1"].shape == (64, 64)
    assert d1["IMG_TIR1"].shape == (64, 64)
    assert meta["observation_date"] == "2026-08-28"
