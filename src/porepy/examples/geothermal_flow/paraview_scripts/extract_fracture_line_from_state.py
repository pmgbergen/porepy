from __future__ import annotations

import argparse
from pathlib import Path

import paraview
from paraview.simple import *
import builtins



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True)
    parser.add_argument("--pvd", required=True)
    parser.add_argument("--out-dir", required=True)

    # Either --times-days OR --time-index should be provided.
    parser.add_argument("--times-days", nargs="+", type=float, default=None)
    parser.add_argument("--time-index", type=int, default=None)

    parser.add_argument("--reader-index", type=int, default=0)
    parser.add_argument("--source-index", type=int, default=0)
    parser.add_argument(
        "--source-name",
        default=None,
        help="Name of the ParaView source to export, e.g. Threshold1 or CellCenters1.",
    )
    parser.add_argument(
        "--field-association",
        default="Point Data",
        choices=["Point Data", "Cell Data", "Field Data"],
        help="Data association to export to CSV.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help=(
            "Optional fixed output CSV filename. If provided and only one time is "
            "exported, this name is used instead of fracture_profile_t_*_days.csv."
        ),
    )
    return parser.parse_args()


def get_xml_name(source) -> str:
    try:
        return source.GetXMLName()
    except Exception:
        return "<unknown>"


def print_sources() -> None:
    print("Available sources:")
    for i, (key, source) in enumerate(GetSources().items()):
        print(f"  [{i}] {key}: {get_xml_name(source)}")


def sources_by_xml_name(xml_name: str):
    found = []
    for key, source in GetSources().items():
        if get_xml_name(source) == xml_name:
            found.append((key, source))
    return found


def source_by_name(source_name: str):
    for key, source in GetSources().items():
        # key is usually like ('Threshold1', '1234')
        if isinstance(key, tuple) and key[0] == source_name:
            return key, source
        if str(key) == source_name:
            return key, source
    return None


def choose_export_source(source_name: str | None, source_index: int):
    if source_name is not None:
        selected = source_by_name(source_name)
        if selected is None:
            print_sources()
            raise RuntimeError(f"No source named {source_name!r} found in the state.")
        key, source = selected
        print(f"Using export source by name: {key}: {get_xml_name(source)}")
        return source

    preferred_xml_names = [
        "Threshold",
        "PlotOverLine",
        "ExtractBlock",
        "CellDatatoPointData",
        "Calculator",
        "CellCenters",
    ]

    candidates = []
    for xml_name in preferred_xml_names:
        candidates.extend(sources_by_xml_name(xml_name))

    if not candidates:
        print_sources()
        raise RuntimeError("No suitable export source found.")

    print("CSV export candidates:")
    for i, (key, source) in enumerate(candidates):
        print(f"  [{i}] {key}: {get_xml_name(source)}")

    if source_index < 0 or source_index >= len(candidates):
        raise IndexError(
            f"source-index {source_index} out of range for "
            f"{len(candidates)} candidate(s)."
        )

    key, source = candidates[source_index]
    print(f"Using export source [{source_index}]: {key}: {get_xml_name(source)}")
    return source


def closest_time(target: float, available: list[float]) -> float:
    return builtins.min(available, key=lambda t: abs(float(t) - target))


def main():
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

    LoadState(str(state_path))

    pvd_readers = sources_by_xml_name("PVDReader")
    if not pvd_readers:
        print_sources()
        raise RuntimeError("No PVDReader found in state.")

    if args.reader_index < 0 or args.reader_index >= len(pvd_readers):
        raise IndexError(
            f"reader-index {args.reader_index} out of range for "
            f"{len(pvd_readers)} PVD reader(s)."
        )

    _, reader = pvd_readers[args.reader_index]
    reader.FileName = str(pvd_path)
    reader.UpdatePipeline()

    export_source = choose_export_source(args.source_name, args.source_index)

    animation_scene = GetAnimationScene()
    animation_scene.UpdateAnimationUsingDataTimeSteps()

    available_times = list(reader.TimestepValues)
    if not available_times:
        raise RuntimeError("No time steps found in PVD reader.")
    

    jobs = []

    if args.times_days is not None:
        for time_days in args.times_days:
            requested_seconds = time_days * 24.0 * 3600.0
            selected_time = closest_time(requested_seconds, available_times)
            label_days = time_days
            jobs.append((selected_time, label_days))

    elif args.time_index is not None:
        selected_time = available_times[args.time_index]
        label_days = selected_time / (24.0 * 3600.0)
        jobs.append((selected_time, label_days))

    else:
        raise ValueError("Provide either --times-days or --time-index.")

    for selected_time, label_days in jobs:
        animation_scene.AnimationTime = selected_time
        reader.UpdatePipeline(time=selected_time)
        export_source.UpdatePipeline(time=selected_time)

        if args.output_name is not None and len(jobs) == 1:
            out_file = out_dir / args.output_name
        else:
            label = f"{label_days:.6g}"
            out_file = out_dir / f"fracture_profile_t_{label}_days.csv"

        SaveData(
            str(out_file),
            proxy=export_source,
            FieldAssociation=args.field_association,
        )

        print(
            f"Saved {out_file} using PVD time {selected_time:g} seconds "
            f"with FieldAssociation={args.field_association!r}"
        )
    # for time_days in args.times_days:
    #     requested_seconds = time_days * 24.0 * 3600.0
    #     selected_time = closest_time(requested_seconds, available_times)

    #     animation_scene.AnimationTime = selected_time
    #     reader.UpdatePipeline(time=selected_time)
    #     export_source.UpdatePipeline(time=selected_time)

    #     out_file = out_dir / f"fracture_profile_t_{time_days:g}_days.csv"

    #     SaveData(
    #         str(out_file), 
    #         proxy=export_source,
    #         FieldAssociation="Point Data",
    #     )

    #     print(
    #         f"Saved {out_file} "
    #         f"(requested {time_days:g} days, used {selected_time:g} seconds)"
    #     )


if __name__ == "__main__":
    main()