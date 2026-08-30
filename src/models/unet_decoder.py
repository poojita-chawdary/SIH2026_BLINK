"""
Physics-Guided Multi-Scale Refinement U-Net Decoder.
Fuses bidirectional warped frames, optical flow velocity fields, and ConvLSTM
spatiotemporal latent states with adaptive blending masks and residual radiance correction.
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Residual Double 2D Convolution block with GroupNorm/BatchNorm and LeakyReLU.
    """

    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, mid_channels), num_channels=mid_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x) + self.shortcut(x)


class PhysicsGuidedUNetDecoder(nn.Module):
    """
    Multi-Scale Physics-Guided Refinement Decoder for Frame Synthesis.
    Combines bidirectional warped candidates with spatiotemporal latent states
    to generate an artifact-free, ghosting-suppressed synthesis frame.
    """

    def __init__(
        self,
        in_channels: int = 6,  # 2 warped frames (3+3) + optional flow & states
        out_channels: int = 3,
        features: List[int] = [64, 128, 256],
        bilinear: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features

        # Encoder stages
        self.inc = DoubleConv(in_channels, features[0])
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(features[0], features[1]),
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(features[1], features[2]),
        )

        # Bottleneck at (H/4, W/4)
        self.bottleneck = DoubleConv(features[2], features[2])

        # Decoder stages:
        # up1: from (H/4, W/4) -> (H/2, W/2), concatenates with x2 (features[1])
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_up1 = DoubleConv(features[2] + features[1], features[1])

        # up2: from (H/2, W/2) -> (H, W), concatenates with x1 (features[0])
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_up2 = DoubleConv(features[1] + features[0], features[0])

        # Output heads:
        # 1. Soft blending mask M_t in [0, 1] per channel
        # 2. Residual radiance correction delta_I in [-1, 1]
        self.mask_head = nn.Sequential(
            nn.Conv2d(features[0], out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.residual_head = nn.Sequential(
            nn.Conv2d(features[0], out_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(
        self,
        warped_0: torch.Tensor,
        warped_1: torch.Tensor,
        t_normalized: float,
        latent_state: Optional[torch.Tensor] = None,
        flow_01: Optional[torch.Tensor] = None,
        flow_10: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Synthesizes intermediate frame at relative time t_normalized in (0, 1).

        Args:
            warped_0: Warped frame from T_0 -> T_t (B, C, H, W).
            warped_1: Warped frame from T_1 -> T_t (B, C, H, W).
            t_normalized: Scalar relative timestamp in [0.0, 1.0].
            latent_state: Optional ConvLSTM hidden state (B, C_latent, H, W).
            flow_01: Optional optical flow T_0 -> T_1 (B, 2, H, W).
            flow_10: Optional optical flow T_1 -> T_0 (B, 2, H, W).

        Returns:
            synthesized_frame: Synthesized radiance tensor (B, C, H, W) in [0.0, 1.0].
        """
        b, c, orig_h, orig_w = warped_0.shape
        device = warped_0.device

        # Ensure spatial dimensions are divisible by 4 for 2 downsampling stages
        pad_h = (4 - orig_h % 4) % 4
        pad_w = (4 - orig_w % 4) % 4

        if pad_h > 0 or pad_w > 0:
            w0 = F.pad(warped_0, (0, pad_w, 0, pad_h), mode="replicate")
            w1 = F.pad(warped_1, (0, pad_w, 0, pad_h), mode="replicate")
        else:
            w0, w1 = warped_0, warped_1

        _, _, h, w = w0.shape

        # Create time broadcast tensor
        t_tensor = torch.full((b, 1, h, w), fill_value=float(t_normalized), device=device, dtype=w0.dtype)

        # Concatenate inputs
        inputs = [w0, w1, t_tensor]

        if flow_01 is not None and flow_10 is not None:
            f01_scaled = flow_01 * float(t_normalized)
            f10_scaled = flow_10 * float(1.0 - t_normalized)
            if pad_h > 0 or pad_w > 0:
                f01_scaled = F.pad(f01_scaled, (0, pad_w, 0, pad_h), mode="replicate")
                f10_scaled = F.pad(f10_scaled, (0, pad_w, 0, pad_h), mode="replicate")
            inputs.extend([f01_scaled, f10_scaled])

        if latent_state is not None:
            if latent_state.shape[2:] != (h, w):
                latent_state = F.interpolate(latent_state, size=(h, w), mode="bilinear", align_corners=True)
            inputs.append(latent_state)

        feat_in = torch.cat(inputs, dim=1)

        # Project to in_channels if needed
        if feat_in.shape[1] != self.in_channels:
            if not hasattr(self, "_proj_in") or self._proj_in.in_channels != feat_in.shape[1]:
                self._proj_in = nn.Conv2d(feat_in.shape[1], self.in_channels, kernel_size=1).to(device)
            feat_in = self._proj_in(feat_in)

        # Encoder
        x1 = self.inc(feat_in)      # (B, f0, H, W)
        x2 = self.down1(x1)         # (B, f1, H/2, W/2)
        x3 = self.down2(x2)         # (B, f2, H/4, W/4)
        xb = self.bottleneck(x3)    # (B, f2, H/4, W/4)

        # Decoder with skip connections
        u1 = self.up1(xb)           # (B, f2, H/2, W/2)
        u1 = torch.cat([u1, x2], dim=1)
        d1 = self.conv_up1(u1)      # (B, f1, H/2, W/2)

        u2 = self.up2(d1)           # (B, f1, H, W)
        u2 = torch.cat([u2, x1], dim=1)
        d2 = self.conv_up2(u2)      # (B, f0, H, W)

        # Blending mask M_t and radiance residual
        mask = self.mask_head(d2)
        res = self.residual_head(d2) * 0.05  # Bounded residual to preserve physical conservation

        # Adaptive blending with warped candidates
        cand0 = w0[:, : self.out_channels, :, :] if c >= self.out_channels else w0.repeat(1, self.out_channels // c, 1, 1)
        cand1 = w1[:, : self.out_channels, :, :] if c >= self.out_channels else w1.repeat(1, self.out_channels // c, 1, 1)

        # Physics-prior base blend: (1 - t) * w0 + t * w1
        base_blend = (1.0 - float(t_normalized)) * cand0 + float(t_normalized) * cand1
        # Neural adaptive refinement
        adaptive_offset = (mask - 0.5) * (cand0 - cand1)
        synthesized = base_blend + adaptive_offset + res

        # Crop if padded
        if pad_h > 0 or pad_w > 0:
            synthesized = synthesized[:, :, :orig_h, :orig_w]

        return torch.clamp(synthesized, 0.0, 1.0)
