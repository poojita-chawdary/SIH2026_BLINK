"""
Integration tests for AeroInterpolator pipeline.
"""

import pytest
import torch

from src.ingestion.mosdac_parser import SyntheticMOSDACSimulator
from src.pipeline.interpolator import AeroInterpolator


def test_aero_interpolator_end_to_end():
    interpolator = AeroInterpolator(device="cpu", channels=["IMG_VIS", "IMG_WV", "IMG_TIR1"])

    # Generate 2 simulated observations
    d0 = SyntheticMOSDACSimulator.generate_cyclone_frame(grid_size=(64, 64), t_normalized=0.0)
    d1 = SyntheticMOSDACSimulator.generate_cyclone_frame(grid_size=(64, 64), t_normalized=1.0)

    t0 = interpolator.parser.to_normalized_tensor(d0, device="cpu")
    t1 = interpolator.parser.to_normalized_tensor(d1, device="cpu")

    sub_timesteps = [0.25, 0.5, 0.75]
    result = interpolator.interpolate(t0, t1, sub_timesteps=sub_timesteps)

    assert len(result.synthesized_frames) == 3
    assert len(result.linear_blends) == 3
    for frame in result.synthesized_frames:
        assert frame.shape == (1, 3, 64, 64)
        assert frame.min() >= 0.0
        assert frame.max() <= 1.0

    assert result.mean_latency_ms > 0.0
    assert result.fluid_divergence >= 0.0
