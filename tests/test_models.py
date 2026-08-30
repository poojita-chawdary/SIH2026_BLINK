"""
Unit tests for RAFT Optical Flow, Backward Warping, ConvLSTM, and U-Net decoder.
"""

import pytest
import torch

from src.models.conv_lstm import ConvLSTM, ConvLSTMCell
from src.models.raft_engine import RAFTEngine, backward_warp
from src.models.unet_decoder import PhysicsGuidedUNetDecoder


def test_backward_warp_translation():
    # Test that shifting an image by +10px in x and backward warping with dx=-10 recovers the original
    b, c, h, w = 1, 1, 64, 64
    img = torch.zeros(b, c, h, w)
    img[:, :, 20:40, 20:40] = 1.0  # Square pattern

    # Flow field: sample from (x - 5, y)
    flow = torch.zeros(b, 2, h, w)
    flow[:, 0, :, :] = -5.0  # dx = -5

    warped = backward_warp(img, flow, align_corners=True)
    assert warped.shape == (b, c, h, w)
    # The active square in warped should have shifted right by 5 pixels (now at 25:45)
    assert warped[:, :, 20:40, 25:45].mean() > 0.8


def test_raft_engine_bidirectional_flow():
    raft = RAFTEngine(model_type="raft_small", pretrained=False, iters=4, device="cpu")
    f0 = torch.rand(1, 3, 64, 64)
    f1 = torch.rand(1, 3, 64, 64)

    flow_01, flow_10 = raft.estimate_bidirectional_flow(f0, f1)
    assert flow_01.shape == (1, 2, 64, 64)
    assert flow_10.shape == (1, 2, 64, 64)


def test_conv_lstm_forward():
    cell = ConvLSTMCell(input_dim=6, hidden_dim=16, kernel_size=(3, 3))
    x = torch.rand(1, 6, 32, 32)
    h, c = cell(x)
    assert h.shape == (1, 16, 32, 32)
    assert c.shape == (1, 16, 32, 32)

    # Multi-layer ConvLSTM
    lstm = ConvLSTM(input_dim=6, hidden_dims=[16, 8], num_layers=2, batch_first=True)
    seq_x = torch.rand(1, 3, 6, 32, 32)  # (B, T, C, H, W)
    out, states = lstm(seq_x)
    assert out.shape == (1, 3, 8, 32, 32)
    assert len(states) == 2


def test_unet_decoder_synthesis():
    decoder = PhysicsGuidedUNetDecoder(in_channels=13, out_channels=3, features=[16, 32, 64])
    w0 = torch.rand(1, 3, 64, 64)
    w1 = torch.rand(1, 3, 64, 64)
    flow01 = torch.rand(1, 2, 64, 64)
    flow10 = torch.rand(1, 2, 64, 64)
    latent = torch.rand(1, 16, 64, 64)

    synth = decoder(
        warped_0=w0,
        warped_1=w1,
        t_normalized=0.5,
        latent_state=latent,
        flow_01=flow01,
        flow_10=flow10,
    )
    assert synth.shape == (1, 3, 64, 64)
    assert synth.min() >= 0.0
    assert synth.max() <= 1.0
