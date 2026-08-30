"""
Physics-Guided Evaluation & Ground-Truth Verification Suite.
Calculates Peak Signal-to-Noise Ratio (PSNR), Structural Similarity (SSIM),
Optical Flow End-Point Error (EPE), Fluid Divergence Penalty, and Radiance Conservation.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class MetricReport:
    """
    Validation metric report comparing synthesized frames against ground truth.
    """
    psnr_db: float
    ssim: float
    fluid_divergence: float
    radiance_conservation_pct: float
    epe_pixels: Optional[float] = None
    ghosting_reduction_pct: float = 0.0
    inference_latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Union[float, Optional[float]]]:
        return {
            "psnr_db": round(self.psnr_db, 2),
            "ssim": round(self.ssim, 4),
            "fluid_divergence": round(self.fluid_divergence, 6),
            "radiance_conservation_pct": round(self.radiance_conservation_pct, 2),
            "epe_pixels": round(self.epe_pixels, 3) if self.epe_pixels is not None else None,
            "ghosting_reduction_pct": round(self.ghosting_reduction_pct, 2),
            "inference_latency_ms": round(self.inference_latency_ms, 2),
        }


class PhysicsEvaluator:
    """
    Calculates atmospheric physics fidelity and computer vision accuracy metrics.
    """

    @staticmethod
    def compute_psnr(
        pred: torch.Tensor,
        target: torch.Tensor,
        max_val: float = 1.0,
    ) -> float:
        """
        Computes Peak Signal-to-Noise Ratio (dB).
        """
        mse = F.mse_loss(pred, target).item()
        if mse <= 1e-10:
            return 100.0
        return float(10.0 * np.log10((max_val ** 2) / mse))

    @staticmethod
    def compute_ssim(
        pred: torch.Tensor,
        target: torch.Tensor,
        window_size: int = 11,
        max_val: float = 1.0,
    ) -> float:
        """
        Computes Structural Similarity Index (SSIM) across spatial windows.
        """
        c = pred.shape[1]
        device = pred.device

        # 1D Gaussian kernel
        sigma = 1.5
        gauss_1d = torch.tensor(
            [np.exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2)) for x in range(window_size)],
            dtype=torch.float32,
            device=device,
        )
        gauss_1d = gauss_1d / gauss_1d.sum()

        # 2D Gaussian filter
        gauss_2d = (gauss_1d.unsqueeze(1) @ gauss_1d.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
        kernel = gauss_2d.repeat(c, 1, 1, 1)

        pad = window_size // 2
        mu1 = F.conv2d(pred, kernel, padding=pad, groups=c)
        mu2 = F.conv2d(target, kernel, padding=pad, groups=c)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(pred * pred, kernel, padding=pad, groups=c) - mu1_sq
        sigma2_sq = F.conv2d(target * target, kernel, padding=pad, groups=c) - mu2_sq
        sigma12 = F.conv2d(pred * target, kernel, padding=pad, groups=c) - mu1_mu2

        c1 = (0.01 * max_val) ** 2
        c2 = (0.03 * max_val) ** 2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
            (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        )
        return float(ssim_map.mean().item())

    @staticmethod
    def compute_fluid_divergence(flow: torch.Tensor) -> float:
        """
        Calculates fluid divergence penalty ||div(u)||^2 = || du_x/dx + du_y/dy ||^2.

        Args:
            flow: Optical flow velocity field (B, 2, H, W).

        Returns:
            Scalar mean squared divergence.
        """
        # Central difference kernels for spatial derivatives
        device = flow.device
        dx_kernel = torch.tensor([[-0.5, 0.0, 0.5]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        dy_kernel = torch.tensor([[-0.5], [0.0], [0.5]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

        u_x = flow[:, 0:1, :, :]  # Horizontal velocity component
        u_y = flow[:, 1:2, :, :]  # Vertical velocity component

        du_dx = F.conv2d(u_x, dx_kernel, padding=(0, 1))
        du_dy = F.conv2d(u_y, dy_kernel, padding=(1, 0))

        divergence = du_dx + du_dy
        div_loss = torch.mean(divergence ** 2).item()
        return float(div_loss)

    @staticmethod
    def compute_radiance_conservation(
        pred: torch.Tensor,
        frame_0: torch.Tensor,
        frame_1: torch.Tensor,
        t_normalized: float,
    ) -> float:
        """
        Calculates percentage radiance mass conservation against conservative linear bounds.
        """
        pred_mass = pred.mean().item()
        expected_mass = (1.0 - t_normalized) * frame_0.mean().item() + t_normalized * frame_1.mean().item()

        if expected_mass < 1e-6:
            return 100.0

        rel_error = abs(pred_mass - expected_mass) / expected_mass
        conservation_pct = max(0.0, (1.0 - rel_error) * 100.0)
        return float(conservation_pct)

    @staticmethod
    def compute_epe(pred_flow: torch.Tensor, target_flow: torch.Tensor) -> float:
        """
        Computes Mean End-Point Error (EPE) in pixels.
        """
        diff = pred_flow - target_flow
        epe = torch.sqrt(torch.sum(diff ** 2, dim=1) + 1e-6).mean().item()
        return float(epe)

    @classmethod
    def evaluate_synthesis(
        cls,
        synthesized: torch.Tensor,
        ground_truth: torch.Tensor,
        frame_0: torch.Tensor,
        frame_1: torch.Tensor,
        t_normalized: float,
        flow: Optional[torch.Tensor] = None,
        latency_ms: float = 0.0,
    ) -> MetricReport:
        """
        Runs comprehensive quantitative evaluation comparing BLINK synthesis with Ground Truth
        and calculates ghosting reduction relative to linear frame blending.
        """
        psnr_val = cls.compute_psnr(synthesized, ground_truth)
        ssim_val = cls.compute_ssim(synthesized, ground_truth)
        conservation_val = cls.compute_radiance_conservation(synthesized, frame_0, frame_1, t_normalized)

        div_val = cls.compute_fluid_divergence(flow) if flow is not None else 0.0

        # Linear blend baseline for comparison
        linear_blend = (1.0 - t_normalized) * frame_0 + t_normalized * frame_1
        linear_psnr = cls.compute_psnr(linear_blend, ground_truth)

        # Ghosting reduction percentage (based on PSNR gain)
        if linear_psnr < psnr_val:
            ghosting_red = min(100.0, ((psnr_val - linear_psnr) / linear_psnr) * 100.0)
        else:
            ghosting_red = 0.0

        return MetricReport(
            psnr_db=psnr_val,
            ssim=ssim_val,
            fluid_divergence=div_val,
            radiance_conservation_pct=conservation_val,
            ghosting_reduction_pct=ghosting_red,
            inference_latency_ms=latency_ms,
        )
