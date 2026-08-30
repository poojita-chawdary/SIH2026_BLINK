import argparse
import sys
from pathlib import Path

# Add project root
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.ingestion.mosdac_parser import SyntheticMOSDACSimulator


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic INSAT-3DS HDF5 observations.")
    parser.add_argument("--output_dir", type=str, default="data/raw_netcdf", help="Directory to save HDF5 files")
    parser.add_argument("--scenario", type=str, default="cyclone", choices=["cyclone", "cloudburst"], help="Atmospheric scenario")
    parser.add_argument("--resolution", type=int, default=512, help="Image tile resolution (e.g., 512)")
    args = parser.parse_args()

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"🛰️ Generating synthetic INSAT-3DS Multi-Spectral HDF5 files for scenario: {args.scenario}...")

    # Generate T_0 (0 min) and T_1 (15 min)
    if args.scenario == "cloudburst":
        data_0 = SyntheticMOSDACSimulator.generate_convective_cloudburst_frame(grid_size=(args.resolution, args.resolution), t_normalized=0.0)
        data_1 = SyntheticMOSDACSimulator.generate_convective_cloudburst_frame(grid_size=(args.resolution, args.resolution), t_normalized=1.0)
    else:
        data_0 = SyntheticMOSDACSimulator.generate_cyclone_frame(grid_size=(args.resolution, args.resolution), t_normalized=0.0)
        data_1 = SyntheticMOSDACSimulator.generate_cyclone_frame(grid_size=(args.resolution, args.resolution), t_normalized=1.0)

    file_0 = out_path / f"3DIMG_14AUG2026_0000_{args.scenario.upper()}_L1B_STD.h5"
    file_1 = out_path / f"3DIMG_14AUG2026_0015_{args.scenario.upper()}_L1B_STD.h5"

    SyntheticMOSDACSimulator.save_to_hdf5(file_0, data_0, timestamp_str="2026-08-14T00:00:00Z")
    SyntheticMOSDACSimulator.save_to_hdf5(file_1, data_1, timestamp_str="2026-08-14T00:15:00Z")

    print(f"✅ Generated T_0 file: {file_0} ({file_0.stat().st_size / 1024:.1f} KB)")
    print(f"✅ Generated T_1 file: {file_1} ({file_1.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
