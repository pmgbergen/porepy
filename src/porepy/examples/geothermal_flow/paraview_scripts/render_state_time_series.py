"""Render a ParaView state at multiple times.

This is useful for figures such as:

1. The 2x2 s_halite evolution plot:
   t = 1, 10, 30, 74 days.

2. The Figure 8 PHZ column plot:
   one 3-row ParaView layout rendered at t = 10 and 74 days.

Example
-------
pvbatch geothermal_flow/paraview_scripts/render_state_time_series.py \
    --state geothermal_flow/paraview_states/example1/s_halite_panel.pvsm \
    --pvd visualization/example1/example1.pvd \
    --out-dir figures/example1/s_halite_panels \
    --times-days 1 10 30 74 \
    --output-prefix s_halite \
    --width 1580 \
    --height 840 \
    --save-layout
"""

from __future__ import annotations

import argparse
import builtins
from pathlib import Path

import paraview
from paraview.simple import *


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--state", required=True, help="Path to the ParaView .pvsm state."
    )
    parser.add_argument(
        "--pvd", required=True, help="Path to the simulation .pvd file."
    )
    parser.add_argument(
        "--out-dir", required=True, help="Directory for panel PNG files."
    )
    parser.add_argument("--times-days", nargs="+", type=float, required=True)
    parser.add_argument("--width", type=int, default=1580)
    parser.add_argument("--height", type=int, default=840)
    parser.add_argument("--reader-index", type=int, default=0)
    parser.add_argument(
        "--output-prefix",
        default="s_halite",
        help="Prefix used for saved PNG files.",
    )
    parser.add_argument(
        "--save-layout",
        action="store_true",
        help="Save the full ParaView layout instead of only the active render view.",
    )
    parser.add_argument(
        "--update-all-pvd-readers",
        action="store_true",
        help="Update all PVD readers in the state. Useful for multi-source states.",
    )
    return parser.parse_args()


def get_xml_name(source) -> str:
    """Return ParaView XML proxy name safely."""
    try:
        return source.GetXMLName()
    except Exception:
        return "<unknown>"


def find_pvd_readers() -> list[tuple[object, object]]:
    """Find all PVDReader sources in the loaded state."""
    readers = []
    for key, source in GetSources().items():
        if get_xml_name(source) == "PVDReader":
            readers.append((key, source))
    return readers


def closest_time(target: float, available: list[float]) -> float:
    """Return available PVD time closest to target time."""
    return builtins.min(available, key=lambda value: abs(float(value) - target))


def set_time(reader, selected_time: float) -> None:
    """Set animation and all views to a selected simulation time."""
    scene = GetAnimationScene()
    scene.AnimationTime = selected_time

    reader.UpdatePipeline(time=selected_time)

    for view in GetViews():
        if hasattr(view, "ViewTime"):
            view.ViewTime = selected_time


def load_state(state_path: Path) -> None:
    """Load a ParaView state file.

    ``LoadStateDataFileOptions='Use File Names From State'`` is compatible with
    ParaView 6.1. The PVD reader filenames are replaced immediately after load.
    """
    try:
        LoadState(str(state_path), LoadStateDataFileOptions="Use File Names From State")
    except Exception as exc:
        print(
            "Warning: LoadStateDataFileOptions was not accepted; "
            f"falling back to plain LoadState. Original error: {exc}"
        )
        LoadState(str(state_path))


def save_screenshot(
    out_file: Path,
    *,
    save_layout: bool,
    width: int,
    height: int,
) -> None:
    """Save either the full ParaView layout or the active render view."""
    if save_layout:
        target = GetLayout()
    else:
        target = GetActiveViewOrCreate("RenderView")

    SaveScreenshot(
        str(out_file),
        target,
        ImageResolution=[width, height],
        TransparentBackground=0,
    )


def main() -> None:
    """Load state, swap PVD file, render requested time snapshots."""
    args = parse_args()

    state_path = Path(args.state).resolve()
    pvd_path = Path(args.pvd).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not state_path.exists():
        raise FileNotFoundError(f"State file not found: {state_path}")
    if not pvd_path.exists():
        raise FileNotFoundError(f"PVD file not found: {pvd_path}")

    paraview.simple._DisableFirstRenderCameraReset()

    load_state(state_path)

    readers = find_pvd_readers()
    if not readers:
        print("Available sources:")
        for key, source in GetSources().items():
            print(f"  {key}: {get_xml_name(source)}")
        raise RuntimeError("No PVDReader found in loaded state.")

    if args.reader_index < 0 or args.reader_index >= len(readers):
        raise IndexError(f"reader-index {args.reader_index} out of range.")

    if args.update_all_pvd_readers:
        readers_to_update = readers
    else:
        readers_to_update = [readers[args.reader_index]]

    for key, reader in readers_to_update:
        print(f"Replacing {key} with {pvd_path}")
        reader.FileName = str(pvd_path)
        reader.UpdatePipeline()

    reader_for_time = readers[args.reader_index][1]

    scene = GetAnimationScene()
    scene.UpdateAnimationUsingDataTimeSteps()

    available_times = list(reader_for_time.TimestepValues)
    if not available_times:
        raise RuntimeError("No time steps found in PVD reader.")

    for time_days in args.times_days:
        requested_seconds = time_days * 24.0 * 3600.0
        selected_time = closest_time(requested_seconds, available_times)

        set_time(reader_for_time, selected_time)
        RenderAllViews()

        safe_label = f"{time_days:g}".replace(".", "p")
        out_file = out_dir / f"{args.output_prefix}_t_{safe_label}_days.png"

        save_screenshot(
            out_file,
            save_layout=args.save_layout,
            width=args.width,
            height=args.height,
        )

        print(
            f"Saved {out_file} "
            f"(requested {time_days:g} days, used {selected_time:g} seconds)"
        )


if __name__ == "__main__":
    main()
