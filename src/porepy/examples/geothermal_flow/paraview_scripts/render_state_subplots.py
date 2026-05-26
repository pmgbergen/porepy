"""Render a ParaView state file using a replacement PVD file.

Supports both single-view and multi-panel/subplot ParaView states.

Example for subplot layout:

    pvbatch geothermal_flow/paraview_scripts/render_state_subplots.py \
        --state geothermal_flow/paraview_states/example1/hal_sat_panel.pvsm \
        --pvd visualization/example1/example1.pvd \
        --out figures/example1/vap_hal_sat.png \
        --save-layout \
        --update-all-pvd-readers
"""

from __future__ import annotations

import argparse
from pathlib import Path

import paraview
from paraview.simple import *


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Render a ParaView state file with a replacement PVD file."
    )
    parser.add_argument("--state", required=True)
    parser.add_argument("--pvd", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--width", type=int, default=1580)
    parser.add_argument("--height", type=int, default=840)
    parser.add_argument("--time-index", type=int, default=-1)
    parser.add_argument(
        "--save-layout",
        action="store_true",
        help="Save the full ParaView layout instead of one render view.",
    )
    parser.add_argument(
        "--layout-name",
        default=None,
        help="Name of the layout to save. If omitted, use layout-index.",
    )
    parser.add_argument(
        "--layout-index",
        type=int,
        default=0,
        help="Index of layout to save when --save-layout is used.",
    )
    parser.add_argument("--reader-index", type=int, default=0)
    parser.add_argument(
        "--update-all-pvd-readers",
        action="store_true",
        help="Replace all PVD readers in the state with the given PVD.",
    )
    return parser.parse_args()


def xml_name(proxy: object) -> str:
    """Return ParaView XML name safely."""
    try:
        return proxy.GetXMLName()
    except Exception:
        return "<unknown>"


def get_layout_views(layout: object) -> list[object]:
    """Return views contained in a layout."""
    try:
        return list(GetViewsInLayout(layout))
    except Exception:
        views = []
        for view in GetViews():
            try:
                location = layout.GetViewLocation(view)
                if location != -1:
                    views.append(view)
            except Exception:
                pass
        return views


def print_layouts_and_views() -> None:
    """Print layout and view diagnostics."""
    layouts = GetLayouts()

    print("Available layouts:")
    for i, (name, layout) in enumerate(layouts.items()):
        layout_views = get_layout_views(layout)
        print(f"  [{i}] {name}: {layout} | views={len(layout_views)}")
        for j, view in enumerate(layout_views):
            print(f"      view[{j}] {xml_name(view)}: {view}")

    print("All views:")
    for i, view in enumerate(GetViews()):
        print(f"  [{i}] {xml_name(view)}: {view}")


def select_layout(layout_name: str | None, layout_index: int) -> object:
    """Select a layout by name or index."""
    layouts = GetLayouts()

    if not layouts:
        raise RuntimeError("No ParaView layouts found in loaded state.")

    if layout_name is not None:
        for name, layout in layouts.items():
            if name == layout_name:
                print(f"Selected layout by name: {name}")
                return layout

        print_layouts_and_views()
        raise RuntimeError(f"No layout named {layout_name!r} found.")

    layout_items = list(layouts.items())

    if layout_index < 0 or layout_index >= len(layout_items):
        print_layouts_and_views()
        raise IndexError(
            f"layout-index {layout_index} out of range for "
            f"{len(layout_items)} layout(s)."
        )

    name, layout = layout_items[layout_index]
    print(f"Selected layout by index {layout_index}: {name}")
    return layout


def find_pvd_readers() -> list[tuple[object, object]]:
    """Return all PVDReader sources currently loaded in ParaView."""
    pvd_readers: list[tuple[object, object]] = []

    for source_key, source in GetSources().items():
        if xml_name(source) == "PVDReader":
            pvd_readers.append((source_key, source))

    return pvd_readers


def print_available_sources() -> None:
    """Print loaded sources for debugging."""
    print("Available sources:")
    for source_key, source in GetSources().items():
        print(f"  {source_key}: {xml_name(source)}")


def replace_pvd_files(
    pvd_readers: list[tuple[object, object]],
    pvd_path: Path,
    reader_index: int,
    update_all: bool,
) -> object:
    """Replace PVD reader paths and return reader used for time selection."""
    if not pvd_readers:
        print_available_sources()
        raise RuntimeError("No PVDReader found in the loaded state.")

    if reader_index < 0 or reader_index >= len(pvd_readers):
        raise IndexError(
            f"reader-index {reader_index} out of range for "
            f"{len(pvd_readers)} PVD reader(s)."
        )

    print("PVD readers found:")
    for i, (source_key, reader) in enumerate(pvd_readers):
        print(f"  [{i}] {source_key}: {reader}")

    if update_all:
        readers_to_update = pvd_readers
        print("Updating all PVD readers.")
    else:
        readers_to_update = [pvd_readers[reader_index]]
        print(f"Updating only PVD reader index {reader_index}.")

    for source_key, reader in readers_to_update:
        print(f"Replacing reader {source_key} FileName with: {pvd_path}")
        reader.FileName = str(pvd_path)
        reader.UpdatePipeline()

    return pvd_readers[reader_index][1]


def set_render_time(reader: object, time_index: int) -> None:
    """Set animation and view time from reader timesteps."""
    animation_scene = GetAnimationScene()
    animation_scene.UpdateAnimationUsingDataTimeSteps()

    try:
        time_values = list(reader.TimestepValues)
    except Exception:
        time_values = []

    if not time_values:
        print("No time steps found in the PVD reader; keeping state time.")
        return

    if time_index < -len(time_values) or time_index >= len(time_values):
        raise IndexError(
            f"time-index {time_index} is out of range for "
            f"{len(time_values)} time step(s)."
        )

    selected_time = time_values[time_index]
    animation_scene.AnimationTime = selected_time

    for view in GetViews():
        if hasattr(view, "ViewTime"):
            view.ViewTime = selected_time

    print(f"Using time index {time_index}: {selected_time}")


def save_screenshot(
    out_path: Path,
    width: int,
    height: int,
    save_layout: bool,
    layout_name: str | None,
    layout_index: int,
) -> None:
    """Save either a full layout or a single render view."""
    RenderAllViews()

    print_layouts_and_views()

    if save_layout:
        target = select_layout(layout_name, layout_index)
        print("Saving full ParaView layout.")
    else:
        render_views = [view for view in GetViews() if xml_name(view) == "RenderView"]
        if render_views:
            target = render_views[0]
            print("Saving first available render view.")
        else:
            target = GetActiveViewOrCreate("RenderView")
            print("No render view found; created a new one.")

    SaveScreenshot(
        str(out_path),
        target,
        ImageResolution=[width, height],
        TransparentBackground=0,
    )


def main() -> None:
    """Load state, replace PVD reader, set time, and save screenshot."""
    args = parse_args()

    state_path = Path(args.state).resolve()
    pvd_path = Path(args.pvd).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not state_path.exists():
        raise FileNotFoundError(f"State file not found: {state_path}")
    if not pvd_path.exists():
        raise FileNotFoundError(f"PVD file not found: {pvd_path}")

    print(f"State file: {state_path}")
    print(f"PVD file: {pvd_path}")
    print(f"Output file: {out_path}")

    paraview.simple._DisableFirstRenderCameraReset()

    LoadState(str(state_path))

    pvd_readers = find_pvd_readers()

    reader_for_time = replace_pvd_files(
        pvd_readers=pvd_readers,
        pvd_path=pvd_path,
        reader_index=args.reader_index,
        update_all=args.update_all_pvd_readers,
    )

    set_render_time(reader_for_time, args.time_index)

    save_screenshot(
        out_path=out_path,
        width=args.width,
        height=args.height,
        save_layout=args.save_layout,
        layout_name=args.layout_name,
        layout_index=args.layout_index,
    )

    print(f"Saved figure: {out_path}")


if __name__ == "__main__":
    main()