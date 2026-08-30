"""
Unit tests for physics evaluation and validation metrics.
"""

import numpy as np
import pytest
import torch

from src.pipeline.physics_eval import PhysicsEvaluator


def test_psnr_and_ssim_identical():
    img = torch.rand(1, 3, 64, 64)
    psnr = PhysicsEvaluator.compute_psnr(img, img)
    ssim = PhysicsEvaluator.compute_ssim(img, img)

    assert psnr >= 90.0  # Identical images yield near infinity / capped at 100
    assert abs(ssim - 1.0) < 1e-4


def test_fluid_divergence():
    # Zero flow -> Zero divergence
    zero_flow = torch.zeros(1, 2, 64, 64)
    div_zero = PhysicsEvaluator.compute_fluid_divergence(zero_flow)
    assert div_zero == 0.0

    # Divergent flow field: u_x = x, u_y = y -> du_x/dx = 1, du_y/dy = 1 -> div = 2
    y, x = torch.meshgrid(torch.linspace(-1, 1, 64), torch.linspace(-1, 1, 64), indexing="ij")
    div_flow = torch.stack([x, y], dim=0).unsqueeze(0)
    div_val = PhysicsEvaluator.compute_fluid_divergence(div_flow)
    assert div_val > 0.0


def test_radiance_conservation():
    f0 = torch.ones(1, 3, 32, 32) * 0.4
    f1 = torch.ones(1, 3, 32, 32) * 0.8
    # Exact linear mass at t=0.5 is 0.6
    pred_perfect = torch.ones(1, 3, 32, 32) * 0.6
    cons = PhysicsEvaluator.compute_radiance_conservation(pred_perfect, f0, f1, t_normalized=0.5)
    assert abs(cons - 100.0) < 1e-3
