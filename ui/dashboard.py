"""
Streamlit Operational Dashboard for Project BLINK.
Provides side-by-side comparative split screen: Standard Linear Blend (ghosting demonstration)
vs. BLINK AI Physics-Guided Synthesis with interactive temporal scrubber, spectral band selector,
and real-time metrics telemetry HUD.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np
import torch
from PIL import Image

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

from src.ingestion.mosdac_parser import CHANNEL_CALIBRATION_BOUNDS, SyntheticMOSDACSimulator
from src.ingestion.preprocessor import GeoNormalizer
from src.pipeline.interpolator import AeroInterpolator
from src.pipeline.physics_eval import PhysicsEvaluator


def run_streamlit_app():
    if not STREAMLIT_AVAILABLE:
        print("Streamlit is not installed in this environment. Launching FastAPI console instead.")
        return

    st.set_page_config(
        page_title="BLINK | INSAT-3DS Rapid Scanning Console",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom styling
    st.markdown("""
        <style>
            .main { background-color: #060911; }
            h1, h2, h3 { color: #38bdf8 !important; font-family: 'Space Grotesk', sans-serif; }
            .stMetric { background: #0e1422; border: 1px solid #24304d; border-radius: 8px; padding: 12px; }
        </style>
    """, unsafe_allow_html=True)

    st.title("Project BLINK : Aero-Interpolate Console")
    st.caption("Zero-Payload Rapid Scanning Engine for INSAT-3DS / INSAT-3DR Earth Observation")

    # Sidebar controls
    st.sidebar.header("Simulation & Band Settings")
    scenario = st.sidebar.selectbox(
        "Atmospheric Scenario",
        ["Cyclone Vortex (Rotational Advection)", "Convective Cloudburst (Meso-scale Expansion)"],
    )
    upsample_factor = st.sidebar.select_slider(
        "Temporal Cadence Upsampling",
        options=[3, 5, 15],
        value=15,
        format_func=lambda x: f"{x}x (15m → {15//x}m cadence)",
    )
    band_view = st.sidebar.selectbox(
        "Spectral Channel Display",
        ["RGB False Color (VIS + WV + TIR1)", "IMG_TIR1 (10.8 µm Thermal IR)", "IMG_WV (6.8 µm Water Vapour)", "IMG_VIS (0.65 µm Visible)"],
    )

    # Initialize interpolator
    @st.cache_resource
    def get_interpolator():
        return AeroInterpolator(device="auto", channels=["IMG_VIS", "IMG_WV", "IMG_TIR1"])

    interpolator = get_interpolator()

    # Generate synthetic observations T_0 and T_1
    is_cloudburst = "Cloudburst" in scenario
    if is_cloudburst:
        d0 = SyntheticMOSDACSimulator.generate_convective_cloudburst_frame(t_normalized=0.0)
        d1 = SyntheticMOSDACSimulator.generate_convective_cloudburst_frame(t_normalized=1.0)
    else:
        d0 = SyntheticMOSDACSimulator.generate_cyclone_frame(t_normalized=0.0)
        d1 = SyntheticMOSDACSimulator.generate_cyclone_frame(t_normalized=1.0)

    t0 = interpolator.parser.to_normalized_tensor(d0, device=interpolator.device)
    t1 = interpolator.parser.to_normalized_tensor(d1, device=interpolator.device)

    # Sub-timesteps
    sub_timesteps = [round(i / upsample_factor, 4) for i in range(1, upsample_factor)]
    result = interpolator.interpolate(t0, t1, sub_timesteps=sub_timesteps)

    # Timeline scrubber
    st.markdown("### Temporal Observation Timeline")
    time_index = st.slider(
        "Synthesized Observation Time",
        min_value=0,
        max_value=len(sub_timesteps) + 1,
        value=len(sub_timesteps) // 2 + 1,
        format="Step %d",
    )

    # Determine frame to display
    if time_index == 0:
        cur_t_min = 0.0
        ai_frame = t0
        lin_frame = t0
    elif time_index == len(sub_timesteps) + 1:
        cur_t_min = 15.0
        ai_frame = t1
        lin_frame = t1
    else:
        f_idx = time_index - 1
        cur_t_min = sub_timesteps[f_idx] * 15.0
        ai_frame = result.synthesized_frames[f_idx]
        lin_frame = result.linear_blends[f_idx]

    # Convert to RGB images
    ai_rgb = GeoNormalizer.tensor_to_rgb_preview(ai_frame)
    lin_rgb = GeoNormalizer.tensor_to_rgb_preview(lin_frame)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Linear Frame Blend (Ghosting Artifacts)")
        st.image(lin_rgb, caption=f"Standard Linear Interpolation at T + {cur_t_min:.1f} min", use_container_width=True)
    with col2:
        st.subheader("BLINK Physics-Guided Synthesis (Zero-Ghosting)")
        st.image(ai_rgb, caption=f"BLINK Neural Kinematics at T + {cur_t_min:.1f} min", use_container_width=True)

    # Telemetry HUD
    st.markdown("### Real-Time Telemetry HUD & Physics Validation")
    # Ground truth reference
    t_norm = cur_t_min / 15.0
    if is_cloudburst:
        d_gt = SyntheticMOSDACSimulator.generate_convective_cloudburst_frame(t_normalized=t_norm)
    else:
        d_gt = SyntheticMOSDACSimulator.generate_cyclone_frame(t_normalized=t_norm)
    t_gt = interpolator.parser.to_normalized_tensor(d_gt, device=interpolator.device)

    metrics = PhysicsEvaluator.evaluate_synthesis(
        synthesized=ai_frame.to(interpolator.device),
        ground_truth=t_gt,
        frame_0=t0,
        frame_1=t1,
        t_normalized=t_norm,
        flow=result.flow_01.to(interpolator.device),
        latency_ms=result.mean_latency_ms,
    )

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("PSNR Fidelity", f"{metrics.psnr_db:.1f} dB", delta="Target: ≥ 34.5 dB")
    m2.metric("SSIM Structural", f"{metrics.ssim:.4f}", delta="Target: ≥ 0.940")
    m3.metric("Fluid Divergence", f"{metrics.fluid_divergence:.6f}", delta="||∇ · u||²")
    m4.metric("Mass Conservation", f"{metrics.radiance_conservation_pct:.1f}%", delta="Conservative bound")
    m5.metric("Ghosting Reduction", f"-{metrics.ghosting_reduction_pct:.1f}%", delta="vs Linear Blend")
    m6.metric("Inference Latency", f"{metrics.inference_latency_ms:.1f} ms", delta="per 512x512 tile")


if __name__ == "__main__":
    if STREAMLIT_AVAILABLE:
        run_streamlit_app()
    else:
        print("Streamlit not detected. To view the UI, run `python -m uvicorn src.api.server:app --port 8000` and open http://localhost:8000 in your browser.")
