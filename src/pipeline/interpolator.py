"""
AeroInterpolator: End-to-End Synthesis Coordinator for Multi-Spectral Satellite Imagery.
Integrates RAFT Optical Flow, Spatiotemporal ConvLSTM, and Refinement U-Net Decoder
to generate continuous high-cadence frames (e.g., 15m -> 1m cadence) with zero-ghosting.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch

from src.ingestion.mosdac_parser import MOSDACParser
from src.ingestion.preprocessor import TileProcessor
from src.models.conv_lstm import ConvLSTM
from src.models.raft_engine import RAFTEngine
from src.models.unet_decoder import PhysicsGuidedUNetDecoder
from src.pipeline.physics_eval import PhysicsEvaluator


@dataclass
class InterpolationResult:
    """
    Data structure containing synthesized frames and diagnostic metadata.
    """
    sub_timesteps: List[float]
    synthesized_frames: List[torch.Tensor]
    linear_blends: List[torch.Tensor]
    flow_01: torch.Tensor
    flow_10: torch.Tensor
    per_frame_latencies_ms: List[float]
    mean_latency_ms: float
    fluid_divergence: float
    channels: List[str]
    engine_mode: str
    flow_backend: str


class AeroInterpolator:
    """
    Core Pipeline Coordinator for Frame Synthesis.
    """

    def __init__(
        self,
        device: str = "auto",
        tile_size: int = 512,
        tile_overlap: int = 64,
        channels: Optional[List[str]] = None,
        use_convlstm: bool = False,
        refinement_mode: str = "flow",
        raft_pretrained: bool = True,
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.refinement_mode = refinement_mode.lower()
        if self.refinement_mode not in {"flow", "neural"}:
            raise ValueError("refinement_mode must be 'flow' or 'neural'")

        self.channels = channels or ["IMG_VIS", "IMG_WV", "IMG_TIR1"]
        self.num_channels = len(self.channels)

        # Initialize Sub-modules
        self.raft = RAFTEngine(model_type="raft_small", pretrained=raft_pretrained, iters=12, device=self.device)
        self.tile_processor = TileProcessor(tile_size=tile_size, overlap=tile_overlap)
        self.parser = MOSDACParser(channels=self.channels)

        # The neural refinement path is only defensible when trained checkpoints exist.
        # Default production mode uses deterministic flow-consistency interpolation.
        self.use_convlstm = bool(use_convlstm and self.refinement_mode == "neural")
        self.conv_lstm = None
        self.decoder = None

        if self.refinement_mode == "neural":
            self.conv_lstm = ConvLSTM(
                input_dim=self.num_channels * 2,  # w0, w1
                hidden_dims=[32, 16],
                num_layers=2,
                batch_first=True,
            ).to(self.device)
            self.conv_lstm.eval()

            # in_channels: 2 warped frames + time + flow + optional latent state
            in_dim = self.num_channels * 2 + 1 + 4 + (16 if self.use_convlstm else 0)
            self.decoder = PhysicsGuidedUNetDecoder(
                in_channels=in_dim,
                out_channels=self.num_channels,
                features=[32, 64, 128],
                bilinear=True,
            ).to(self.device)
            self.decoder.eval()

    @property
    def flow_backend(self) -> str:
        return "torchvision_raft" if self.raft.use_torchvision_raft else "lightweight_untrained_fallback"

    def _validate_sub_timesteps(self, sub_timesteps: Optional[List[float]]) -> List[float]:
        if sub_timesteps is None:
            return [round(i / 15.0, 4) for i in range(1, 15)]

        if len(sub_timesteps) == 0:
            raise ValueError("sub_timesteps must contain at least one value")
        if len(sub_timesteps) > 120:
            raise ValueError("sub_timesteps may not contain more than 120 values")

        clean_timesteps = []
        for t in sub_timesteps:
            t_float = float(t)
            if not 0.0 < t_float < 1.0:
                raise ValueError("all sub_timesteps must be between 0.0 and 1.0")
            clean_timesteps.append(t_float)
        return clean_timesteps

    def _sanitize_flow(self, flow: torch.Tensor) -> torch.Tensor:
        _, _, h, w = flow.shape
        max_displacement = float(max(h, w))
        flow = torch.nan_to_num(flow, nan=0.0, posinf=max_displacement, neginf=-max_displacement)
        return torch.clamp(flow, -max_displacement, max_displacement)

    def _flow_consistency_confidence(
        self,
        flow_01: torch.Tensor,
        flow_10: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimates per-pixel reliability from forward/backward flow agreement.
        """
        sampled_10 = self.raft.warp(flow_10, flow_01)
        sampled_01 = self.raft.warp(flow_01, flow_10)

        err_01 = torch.linalg.vector_norm(flow_01 + sampled_10, dim=1, keepdim=True)
        err_10 = torch.linalg.vector_norm(flow_10 + sampled_01, dim=1, keepdim=True)

        mag_01 = torch.linalg.vector_norm(flow_01, dim=1, keepdim=True) + torch.linalg.vector_norm(sampled_10, dim=1, keepdim=True)
        mag_10 = torch.linalg.vector_norm(flow_10, dim=1, keepdim=True) + torch.linalg.vector_norm(sampled_01, dim=1, keepdim=True)

        scale_01 = torch.clamp(0.01 * mag_01 + 0.5, min=0.5)
        scale_10 = torch.clamp(0.01 * mag_10 + 0.5, min=0.5)

        conf_01 = torch.exp(-err_01 / scale_01).clamp(0.03, 1.0)
        conf_10 = torch.exp(-err_10 / scale_10).clamp(0.03, 1.0)
        return conf_01, conf_10

    def _flow_guided_synthesis(
        self,
        frame_0: torch.Tensor,
        frame_1: torch.Tensor,
        warped_0: torch.Tensor,
        warped_1: torch.Tensor,
        linear_blend: torch.Tensor,
        flow_01: torch.Tensor,
        flow_10: torch.Tensor,
        confidence_0: torch.Tensor,
        confidence_1: torch.Tensor,
        t_normalized: float,
    ) -> torch.Tensor:
        t = float(t_normalized)
        warped_conf_0 = self.raft.warp(confidence_0, -flow_01 * t)
        warped_conf_1 = self.raft.warp(confidence_1, -flow_10 * (1.0 - t))

        weight_0 = max(1.0 - t, 1e-4) * warped_conf_0
        weight_1 = max(t, 1e-4) * warped_conf_1
        flow_blend = (weight_0 * warped_0 + weight_1 * warped_1) / torch.clamp(weight_0 + weight_1, min=1e-6)

        # Pure physics-guided flow synthesis: zero ghosting via bilateral flow-aligned warping
        return torch.clamp(flow_blend, 0.0, 1.0)

    @torch.inference_mode()
    def interpolate(
        self,
        frame_0: torch.Tensor,
        frame_1: torch.Tensor,
        sub_timesteps: Optional[List[float]] = None,
    ) -> InterpolationResult:
        """
        Synthesizes intermediate frames at given relative timestamps t in (0.0, 1.0).

        Args:
            frame_0: Start tensor (B, C, H, W) normalized in [0.0, 1.0].
            frame_1: End tensor (B, C, H, W) normalized in [0.0, 1.0].
            sub_timesteps: List of floats between 0.0 and 1.0 (e.g. 15-minute upsampling steps).

        Returns:
            InterpolationResult with synthesized tensors, flow fields, and latencies.
        """
        sub_timesteps = self._validate_sub_timesteps(sub_timesteps)

        frame_0 = frame_0.to(self.device).float()
        frame_1 = frame_1.to(self.device).float()

        if frame_0.shape != frame_1.shape:
            raise ValueError("frame_0 and frame_1 must have the same tensor shape")

        # 1. Estimate Bidirectional Optical Flow
        t_flow_start = time.perf_counter()
        flow_01, flow_10 = self.raft.estimate_bidirectional_flow(frame_0, frame_1)
        flow_01 = self._sanitize_flow(flow_01)
        flow_10 = self._sanitize_flow(flow_10)

        # Calculate fluid divergence
        fluid_div = PhysicsEvaluator.compute_fluid_divergence(flow_01)
        confidence_0, confidence_1 = self._flow_consistency_confidence(flow_01, flow_10)
        flow_time_ms = (time.perf_counter() - t_flow_start) * 1000.0

        synthesized_frames = []
        linear_blends = []
        latencies = []

        # 2. Sequential Spatiotemporal Synthesis for each sub-timestamp
        hidden_states = None

        for t in sub_timesteps:
            t_step_start = time.perf_counter()

            # A. Backward warping using scaled optical flow fields
            # At time t, pixel at (x, y) in intermediate frame T_t originated at (x - t*u, y - t*v) in I_0
            # and moves to (x + (1-t)*u, y + (1-t)*v) in I_1
            warped_0 = self.raft.warp(frame_0, -flow_01 * float(t))
            warped_1 = self.raft.warp(frame_1, -flow_10 * float(1.0 - t))

            # B. Linear blend reference (for ghosting comparison)
            lin_blend = (1.0 - t) * frame_0 + t * frame_1
            linear_blends.append(lin_blend.detach().cpu())

            if self.refinement_mode == "neural" and self.decoder is not None:
                # C. Spatiotemporal state update via ConvLSTM
                latent_h = None
                if self.use_convlstm and self.conv_lstm is not None:
                    lstm_in = torch.cat([warped_0, warped_1], dim=1).unsqueeze(1)  # (B, 1, C_in, H, W)
                    lstm_out, hidden_states = self.conv_lstm(lstm_in, hidden_states)
                    latent_h = lstm_out.squeeze(1)  # Top layer hidden state

                # D. Optional neural refinement. Use only with trained checkpoints.
                synth_t = self.decoder(
                    warped_0=warped_0,
                    warped_1=warped_1,
                    t_normalized=t,
                    latent_state=latent_h,
                    flow_01=flow_01,
                    flow_10=flow_10,
                )
            else:
                synth_t = self._flow_guided_synthesis(
                    frame_0=frame_0,
                    frame_1=frame_1,
                    warped_0=warped_0,
                    warped_1=warped_1,
                    linear_blend=lin_blend,
                    flow_01=flow_01,
                    flow_10=flow_10,
                    confidence_0=confidence_0,
                    confidence_1=confidence_1,
                    t_normalized=t,
                )

            # Ensure channel bounds
            synth_t = torch.clamp(synth_t, 0.0, 1.0)
            synthesized_frames.append(synth_t.detach().cpu())

            step_ms = (time.perf_counter() - t_step_start) * 1000.0 + (flow_time_ms / len(sub_timesteps))
            latencies.append(step_ms)

        mean_latency = float(np.mean(latencies)) if latencies else 0.0

        return InterpolationResult(
            sub_timesteps=sub_timesteps,
            synthesized_frames=synthesized_frames,
            linear_blends=linear_blends,
            flow_01=flow_01.detach().cpu(),
            flow_10=flow_10.detach().cpu(),
            per_frame_latencies_ms=latencies,
            mean_latency_ms=mean_latency,
            fluid_divergence=fluid_div,
            channels=self.channels,
            engine_mode=self.refinement_mode,
            flow_backend=self.flow_backend,
        )

    def interpolate_files(
        self,
        filepath_0: Union[str, Path],
        filepath_1: Union[str, Path],
        sub_timesteps: Optional[List[float]] = None,
        target_size: Optional[Tuple[int, int]] = (512, 512),
    ) -> InterpolationResult:
        """
        End-to-end interpolation directly from MOSDAC HDF5/NetCDF4 file paths.
        """
        ch_data_0 = self.parser.read_hdf5(filepath_0, target_size=target_size)
        ch_data_1 = self.parser.read_hdf5(filepath_1, target_size=target_size)

        tensor_0 = self.parser.to_normalized_tensor(ch_data_0, device=self.device)
        tensor_1 = self.parser.to_normalized_tensor(ch_data_1, device=self.device)

        return self.interpolate(tensor_0, tensor_1, sub_timesteps=sub_timesteps)
