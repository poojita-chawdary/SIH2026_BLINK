"""
Tests for scientifically grounded nowcasting:
- NETRA cloudburst algorithm (Shukla et al., 2017)
- Dvorak ADT & Courtney-Knaff (2009) wind-pressure physics
"""

import numpy as np
import pytest
import torch

from src.pipeline.nowcasting import ConvectiveNowcaster, StormTrackPredictor


def test_dvorak_intensity_estimation():
    # Synthetic convective frame (TIR-1 inverted: cold cloud top = 1.0)
    t1 = torch.zeros((1, 3, 128, 128), dtype=torch.float32)
    # Put an intense storm core at center
    t1[:, 2, 50:78, 50:78] = 0.95  # cold core (~198 K)
    t1[:, 2, 60:68, 60:68] = 0.50  # warm eye (~245 K)

    ci, v_kmh, p_hpa, cat = StormTrackPredictor.estimate_dvorak_intensity(
        tensor_t1=t1,
        flow_mag_max=16.0,
        cx_px=64.0,
        cy_px=64.0,
    )

    assert 2.0 <= ci <= 8.0
    assert 30.0 <= v_kmh <= 300.0
    assert 880.0 <= p_hpa <= 1010.0
    assert "Storm" in cat or "Depression" in cat


def test_netra_cloudburst_nowcasting():
    # T0 and T1 frames
    t0 = torch.zeros((1, 3, 128, 128), dtype=torch.float32)
    t1 = torch.zeros((1, 3, 128, 128), dtype=torch.float32)

    # Convective explosion between T0 and T1
    t0[:, 2, 40:88, 40:88] = 0.30  # ~266 K at T0
    t1[:, 2, 40:88, 40:88] = 0.92  # ~201 K at T1 (severe cooling dT/dt ~ -65 K / 15min)

    flow_01 = torch.zeros((1, 2, 128, 128), dtype=torch.float32)
    flow_01[:, 0] = 3.5

    report = ConvectiveNowcaster.evaluate_convective_risk(t0, t1, flow_01)

    assert report.extreme_rain_probability_pct >= 60.0
    assert report.max_cooling_rate_k_15min > 5.0
    assert len(report.active_threat_clusters) > 0
    top_cluster = report.active_threat_clusters[0]
    assert top_cluster.min_brightness_temp_k < 220.0
    assert top_cluster.estimated_rainfall_mm_hr >= 45.0
