import sys
import time
from pathlib import Path

# Add project root
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch

from src.ingestion.mosdac_parser import SyntheticMOSDACSimulator
from src.pipeline.interpolator import AeroInterpolator
from src.pipeline.physics_eval import PhysicsEvaluator


def run_benchmark(
    scenario: str = "cyclone",
    resolution: int = 512,
    cadence_subdivisions: int = 15,
):
    print("=" * 80)
    print(f"🛰️ RUNNING BLINK RAPID-SCAN BENCHMARK [{scenario.upper()}]")
    print(f"   Native Cadence: 15.0 min | Target Synthesized Cadence: {15.0 / cadence_subdivisions:.1f} min")
    print(f"   Spatial Grid: {resolution} x {resolution} Multi-Spectral Tensors (VIS, WV, TIR1)")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    interpolator = AeroInterpolator(device=device, channels=["IMG_VIS", "IMG_WV", "IMG_TIR1"])

    # 1. Generate Observations at T_0 (0 min) and T_1 (15 min)
    if scenario == "cloudburst":
        d0 = SyntheticMOSDACSimulator.generate_convective_cloudburst_frame(grid_size=(resolution, resolution), t_normalized=0.0)
        d1 = SyntheticMOSDACSimulator.generate_convective_cloudburst_frame(grid_size=(resolution, resolution), t_normalized=1.0)
    else:
        d0 = SyntheticMOSDACSimulator.generate_cyclone_frame(grid_size=(resolution, resolution), t_normalized=0.0)
        d1 = SyntheticMOSDACSimulator.generate_cyclone_frame(grid_size=(resolution, resolution), t_normalized=1.0)

    t0 = interpolator.parser.to_normalized_tensor(d0, device=interpolator.device)
    t1 = interpolator.parser.to_normalized_tensor(d1, device=interpolator.device)

    # 2. Run Interpolation
    sub_timesteps = [round(i / cadence_subdivisions, 4) for i in range(1, cadence_subdivisions)]
    result = interpolator.interpolate(t0, t1, sub_timesteps=sub_timesteps)

    # 3. Evaluate each synthesized step against ground-truth
    psnr_scores = []
    ssim_scores = []
    linear_psnr_scores = []
    conservation_scores = []

    print(f"\n{'Step':<6} | {'Timestamp':<12} | {'PSNR (dB)':<10} | {'SSIM':<8} | {'Lin-PSNR':<10} | {'Ghosting Red.':<14} | {'Latency':<10}")
    print("-" * 80)

    for i, t in enumerate(sub_timesteps):
        # Generate exact ground-truth for time t
        if scenario == "cloudburst":
            d_gt = SyntheticMOSDACSimulator.generate_convective_cloudburst_frame(grid_size=(resolution, resolution), t_normalized=t)
        else:
            d_gt = SyntheticMOSDACSimulator.generate_cyclone_frame(grid_size=(resolution, resolution), t_normalized=t)

        t_gt = interpolator.parser.to_normalized_tensor(d_gt, device=interpolator.device)
        synth_t = result.synthesized_frames[i].to(interpolator.device)
        lin_t = result.linear_blends[i].to(interpolator.device)

        metrics = PhysicsEvaluator.evaluate_synthesis(
            synthesized=synth_t,
            ground_truth=t_gt,
            frame_0=t0,
            frame_1=t1,
            t_normalized=t,
            flow=result.flow_01.to(interpolator.device),
            latency_ms=result.per_frame_latencies_ms[i],
        )

        lin_psnr = PhysicsEvaluator.compute_psnr(lin_t, t_gt)

        psnr_scores.append(metrics.psnr_db)
        ssim_scores.append(metrics.ssim)
        linear_psnr_scores.append(lin_psnr)
        conservation_scores.append(metrics.radiance_conservation_pct)

        time_min = t * 15.0
        print(f"T+{i+1:<4} | T + {time_min:4.1f} min   | {metrics.psnr_db:8.2f} dB | {metrics.ssim:6.4f} | {lin_psnr:8.2f} dB | -{metrics.ghosting_reduction_pct:6.1f}%       | {result.per_frame_latencies_ms[i]:6.1f} ms")

    print("-" * 80)
    mean_psnr = float(np.mean(psnr_scores))
    mean_ssim = float(np.mean(ssim_scores))
    mean_lin_psnr = float(np.mean(linear_psnr_scores))
    mean_cons = float(np.mean(conservation_scores))

    print("\n📈 QUANTITATIVE BENCHMARK SUMMARY & SCORECARD")
    print(f"   • Mean Synthesized PSNR : {mean_psnr:.2f} dB (Target: ≥ 34.5 dB) -> {'[PASSED]' if mean_psnr >= 34.5 else '[WARN]'}")
    print(f"   • Mean Synthesized SSIM : {mean_ssim:.4f} (Target: ≥ 0.9400) -> {'[PASSED]' if mean_ssim >= 0.94 else '[WARN]'}")
    print(f"   • Mean Linear Blend PSNR: {mean_lin_psnr:.2f} dB (Ghosting baseline)")
    print(f"   • PSNR Advantage Gain   : +{mean_psnr - mean_lin_psnr:.2f} dB")
    print(f"   • Radiance Conservation : {mean_cons:.2f}% (Target: ≥ 98.0%) -> {'[PASSED]' if mean_cons >= 98.0 else '[WARN]'}")
    print(f"   • Fluid Divergence Loss : {result.fluid_divergence:.6f}")
    print(f"   • Mean Frame Latency    : {result.mean_latency_ms:.2f} ms")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark(scenario="cyclone")
    run_benchmark(scenario="cloudburst")
