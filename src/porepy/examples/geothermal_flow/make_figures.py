"""Figure-generation driver for the geothermal model paper results.

This module reads the figure manifest in ``geothermal_flow/configs/figures.yaml``
and executes the postprocessing workflow needed to reproduce the benchmark and
fractured-reservoir figures.

Depending on the figure entry, the workflow may:

1. render a saved ParaView state file to PNG,
2. render ParaView time-series panels,
3. extract CSV data from a saved ParaView state,
4. extract benchmark or production time-series data from PVD files,
5. assemble comparison panels with Matplotlib, or
6. generate final publication figures from extracted CSV files.

The simulation outputs are assumed to have been generated beforehand by
``geothermal_flow.simulation_driver``. This module does not run the simulations;
it only performs extraction, rendering, and plotting.

Typical usage from ``src/porepy/examples`` is

    python -m geothermal_flow.make_figures

By default, this uses

    geothermal_flow/configs/figures.yaml

To use a different figure manifest, for example inside the Docker
reproducibility image, pass ``--config`` explicitly:

    python -m geothermal_flow.make_figures \\
        --config geothermal_flow/configs/figures.yaml

To generate selected figures only, use

    python -m geothermal_flow.make_figures \\
        --config geothermal_flow/configs/figures.yaml \\
        --figures figure8 figure9

To inspect the commands without executing them, use

    python -m geothermal_flow.make_figures \\
        --config geothermal_flow/configs/figures.yaml \\
        --dry-run

The generated figures are written to the paths specified in the manifest,
typically under ``figures/``. Intermediate CSV files and diagnostic caches are
written under ``csv/`` and ``output/`` when required.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

from click import command

# import yaml
from .io_utils import load_yaml


def run_command(command: list[str], *, dry_run: bool = False) -> None:
    """Print and optionally execute a shell command."""
    print("\n" + "-" * 80)
    print("Running:")
    print(" ".join(command))
    print("-" * 80)

    if dry_run:
        return

    # subprocess.run(command, check=True)
    result = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if result.stdout:
        print(result.stdout, end="")

    harmless_patterns = [
        "openvkl",
        "vtkSMSettings",
        "ParaView-UserSettings.json",
        "vtkOpenGLState",
        "active textures",
        "Leaked for texture object",
        "bad X server connection",
        "vtkContext2DScalarBarActor",
        "printf format",
        "std::format",
        "deprecated in 6.1.0",
    ]

    stderr_lines = result.stderr.splitlines()
    filtered_stderr = [
        line
        for line in stderr_lines
        if not any(pattern in line for pattern in harmless_patterns)
    ]

    if filtered_stderr:
        print("\n".join(filtered_stderr), file=sys.stderr)

    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode,
            command,
            output=result.stdout,
            stderr=result.stderr,
        )


def bool_flag(command: list[str], flag: str, enabled: bool | None) -> None:
    """Append a boolean CLI flag if enabled."""
    if enabled:
        command.append(flag)


def append_if_present(
    command: list[str],
    flag: str,
    config: dict[str, Any],
    key: str,
) -> None:
    """Append ``flag value`` if ``key`` exists in ``config``."""
    if key in config and config[key] is not None:
        command.extend([flag, str(config[key])])


def append_list(
    command: list[str],
    flag: str,
    values: list[Any] | tuple[Any, ...] | None,
) -> None:
    """Append a repeated/list CLI argument."""
    if values:
        command.append(flag)
        command.extend(str(value) for value in values)


def run_benchmark_profile_extract(
    extract_cfg: dict[str, Any],
    *,
    dry_run: bool = False,
) -> None:
    """Extract PorePy benchmark profile from a PVD file."""
    command = [
        sys.executable,
        extract_cfg["script"],
        "--pvd",
        extract_cfg["pvd"],
        "--out-csv",
        extract_cfg["out_csv"],
    ]

    append_if_present(command, "--time-years", extract_cfg, "time_years")
    append_if_present(command, "--time-days", extract_cfg, "time_days")
    append_if_present(command, "--time-index", extract_cfg, "time_index")
    append_if_present(command, "--block-filter", extract_cfg, "block_filter")
    append_if_present(
        command, "--gas-zero-threshold", extract_cfg, "gas_zero_threshold"
    )
    append_if_present(command, "--pvd-time-unit", extract_cfg, "pvd_time_unit")

    run_command(command, dry_run=dry_run)


def run_benchmark_comparison_plot(
    plot_cfg: dict[str, Any],
    *,
    dry_run: bool = False,
) -> None:
    """Run benchmark CSMP--PorePy comparison plot."""
    command = [
        sys.executable,
        plot_cfg["script"],
        "--porepy-csv",
        plot_cfg["porepy_csv"],
        "--reference-dir",
        plot_cfg["reference_dir"],
        "--out",
        plot_cfg["output"],
    ]

    append_if_present(
        command,
        "--residual-liquid-saturation",
        plot_cfg,
        "residual_liquid_saturation",
    )
    append_if_present(
        command,
        "--simulation-time-years",
        plot_cfg,
        "simulation_time_years",
    )
    append_list(command, "--vl-region", plot_cfg.get("vl_region"))
    append_list(command, "--lh-region", plot_cfg.get("lh_region"))

    if plot_cfg.get("no_annotations", False):
        command.append("--no-annotations")

    run_command(command, dry_run=dry_run)


def run_paraview_layout_png(
    name: str,
    cfg: dict[str, Any],
    pvbatch: str,
    *,
    dry_run: bool = False,
) -> None:
    """Run a ParaView state-to-PNG layout render."""
    command = [
        pvbatch,
        cfg["script"],
        "--state",
        cfg["state"],
        "--pvd",
        cfg["pvd"],
        "--out",
        cfg["out"],
    ]

    append_if_present(command, "--width", cfg, "width")
    append_if_present(command, "--height", cfg, "height")
    append_if_present(command, "--time-index", cfg, "time_index")

    bool_flag(command, "--save-layout", cfg.get("save_layout", False))
    bool_flag(
        command, "--update-all-pvd-readers", cfg.get("update_all_pvd_readers", False)
    )

    print(f"\nGenerating {name} as ParaView layout PNG")
    run_command(command, dry_run=dry_run)


def run_paraview_time_series(
    render_cfg: dict[str, Any],
    pvbatch: str,
    *,
    dry_run: bool = False,
) -> None:
    """Render ParaView PNG panels at multiple requested times."""
    command = [
        pvbatch,
        render_cfg["script"],
        "--state",
        render_cfg["state"],
        "--pvd",
        render_cfg["pvd"],
        "--out-dir",
        render_cfg["out_dir"],
    ]

    append_list(command, "--times-days", render_cfg.get("times_days"))
    append_if_present(command, "--width", render_cfg, "width")
    append_if_present(command, "--height", render_cfg, "height")
    append_if_present(command, "--reader-index", render_cfg, "reader_index")
    append_if_present(command, "--output-prefix", render_cfg, "output_prefix")

    bool_flag(command, "--save-layout", render_cfg.get("save_layout", False))

    bool_flag(
        command,
        "--update-all-pvd-readers",
        render_cfg.get("update_all_pvd_readers", False),
    )

    run_command(command, dry_run=dry_run)


def run_paraview_csv_extract(
    extract_cfg: dict[str, Any],
    pvbatch: str,
    *,
    dry_run: bool = False,
) -> None:
    """Extract CSV data using a saved ParaView state."""
    command = [
        pvbatch,
        extract_cfg["script"],
        "--state",
        extract_cfg["state"],
        "--pvd",
        extract_cfg["pvd"],
        "--out-dir",
        extract_cfg["out_dir"],
    ]

    append_list(command, "--times-days", extract_cfg.get("times_days"))
    append_if_present(command, "--time-index", extract_cfg, "time_index")
    append_if_present(command, "--output-name", extract_cfg, "output_name")
    append_if_present(command, "--reader-index", extract_cfg, "reader_index")
    append_if_present(command, "--source-index", extract_cfg, "source_index")
    append_if_present(command, "--source-name", extract_cfg, "source_name")
    append_if_present(command, "--field-association", extract_cfg, "field_association")

    run_command(command, dry_run=dry_run)


def run_centerline_plot(
    plot_cfg: dict[str, Any],
    *,
    dry_run: bool = False,
) -> None:
    """Run the centerline-profile Matplotlib plotting script."""
    command = [
        sys.executable,
        plot_cfg["script"],
        "--input-csv",
        plot_cfg["input_csv"],
        "--out",
        plot_cfg["output"],
    ]

    append_if_present(command, "--arc-length-col", plot_cfg, "arc_length_col")
    append_if_present(command, "--x-padding", plot_cfg, "x_padding")
    append_if_present(command, "--legend-loc", plot_cfg, "legend_loc")

    smoothing = plot_cfg.get("smoothing", {})
    if smoothing.get("enabled", False):
        command.append("--smooth")
        append_list(command, "--smooth-range", smoothing.get("range"))
        append_list(command, "--smooth-vars", smoothing.get("variables"))

        start_values = smoothing.get("starting_values", {})
        if start_values:
            command.append("--smooth-start")
            command.extend(f"{key}={value}" for key, value in start_values.items())

    run_command(command, dry_run=dry_run)


def run_halite_aperture_plot(
    plot_cfg: dict[str, Any],
    *,
    dry_run: bool = False,
) -> None:
    """Run the fracture halite/aperture Matplotlib plotting script."""
    command = [
        sys.executable,
        plot_cfg["script"],
        "--input-dir",
        plot_cfg["input_dir"],
        "--out",
        plot_cfg["output"],
    ]

    append_list(command, "--times-days", plot_cfg.get("times_days"))

    option_map = {
        "clogging_exponent": "--clogging-exponent",
        "minimum_aperture": "--minimum-aperture",
        "reference_aperture": "--reference-aperture",
        "aperture_ymin": "--aperture-ymin",
        "aperture_ymax": "--aperture-ymax",
        "halite_ymin": "--halite-ymin",
        "halite_ymax": "--halite-ymax",
        "x_col": "--x-col",
        "y_col": "--y-col",
        "producer_x": "--producer-x",
        "producer_y": "--producer-y",
    }

    for key, flag in option_map.items():
        append_if_present(command, flag, plot_cfg, key)

    run_command(command, dry_run=dry_run)


def run_assemble_s_halite_panels(
    plot_cfg: dict[str, Any],
    *,
    dry_run: bool = False,
) -> None:
    """Run the Matplotlib script that assembles s_halite time panels."""
    command = [
        sys.executable,
        plot_cfg["script"],
        "--panel-dir",
        plot_cfg["panel_dir"],
        "--out",
        plot_cfg["output"],
    ]
    append_if_present(command, "--colorbar-label", plot_cfg, "colorbar_label")
    append_if_present(command, "--vmin", plot_cfg, "vmin")
    append_if_present(command, "--vmax", plot_cfg, "vmax")
    append_if_present(command, "--cmap", plot_cfg, "cmap")
    append_if_present(command, "--dpi", plot_cfg, "dpi")

    run_command(command, dry_run=dry_run)


def run_assemble_column_phz(
    plot_cfg: dict[str, Any],
    *,
    dry_run: bool = False,
) -> None:
    """Run the Matplotlib/PIL script that assembles Figure 8 PHZ columns."""
    command = [
        sys.executable,
        plot_cfg["script"],
        "--left",
        plot_cfg["left"],
        "--right",
        plot_cfg["right"],
        "--out",
        plot_cfg["output"],
    ]

    append_if_present(command, "--gap", plot_cfg, "gap")
    append_if_present(command, "--pad", plot_cfg, "pad")
    append_if_present(command, "--background", plot_cfg, "background")

    run_command(command, dry_run=dry_run)


def run_near_well_comparison(
    plot_cfg: dict[str, Any],
    *,
    dry_run: bool = False,
) -> None:
    """Run the Matplotlib script that assembles the near-well comparison."""
    command = [
        sys.executable,
        plot_cfg["script"],
        "--left",
        plot_cfg["left"],
        "--right",
        plot_cfg["right"],
        "--out",
        plot_cfg["output"],
    ]

    labels = plot_cfg.get("labels", {})
    if "left" in labels:
        command.extend(["--left-label", labels["left"]])
    if "right" in labels:
        command.extend(["--right-label", labels["right"]])

    colorbar = plot_cfg.get("colorbar", {})
    if "field" in colorbar:
        command.extend(["--colorbar-label", colorbar["field"]])
    if "vmin" in colorbar:
        command.extend(["--vmin", str(colorbar["vmin"])])
    if "vmax" in colorbar:
        command.extend(["--vmax", str(colorbar["vmax"])])

    run_command(command, dry_run=dry_run)


def run_production_diagnostics(
    fig_cfg: dict[str, Any],
    *,
    dry_run: bool = False,
) -> None:
    """Run production diagnostics plotting from cached diagnostics CSV files.

    This assumes that the diagnostics caches already exist. If not, generate them
    using your production extraction utility first.
    """
    plot_cfg = fig_cfg["matplotlib_plot"]

    command = [
        sys.executable,
        plot_cfg["script"],
        "--phi-0-1",
        plot_cfg["phi_0_1"],
        "--phi-1-0",
        plot_cfg["phi_1_0"],
        "--out",
        plot_cfg["output"],
    ]

    option_map = {
        "t_min": "--t-min",
        "t_max": "--t-max",
        "q_ymin": "--q-ymin",
        "q_ymax": "--q-ymax",
        "energy_ymin": "--energy-ymin",
        "energy_ymax": "--energy-ymax",
    }

    for key, flag in option_map.items():
        append_if_present(command, flag, plot_cfg, key)

    run_command(command, dry_run=dry_run)


def run_matplotlib_plot(plot_cfg: dict[str, Any], *, dry_run: bool = False) -> None:
    """Dispatch Matplotlib plotting based on script name."""
    script_name = Path(plot_cfg["script"]).name

    if script_name == "plot_centerline_profiles.py":
        run_centerline_plot(plot_cfg, dry_run=dry_run)

    elif script_name == "plot_halite_aperture_ratio.py":
        run_halite_aperture_plot(plot_cfg, dry_run=dry_run)

    elif script_name == "assemble_s_halite_panels.py":
        run_assemble_s_halite_panels(plot_cfg, dry_run=dry_run)

    elif script_name == "plot_near_well_comparison.py":
        run_near_well_comparison(plot_cfg, dry_run=dry_run)

    elif script_name == "plot_production_diagnostics.py":
        raise RuntimeError(
            "plot_production_diagnostics.py should be handled by "
            "run_production_diagnostics because it needs two input files."
        )
    elif script_name == "assemble_column_phz.py":
        run_assemble_column_phz(plot_cfg, dry_run=dry_run)

    else:
        raise RuntimeError(f"No Matplotlib dispatcher implemented for {script_name}")


def run_well_timeseries_extraction(
    extract_cfg: dict[str, Any],
    *,
    dry_run: bool = False,
) -> None:
    """Run well time-series extraction for all cases in a production figure."""
    script = extract_cfg["script"]

    for case_name, case_cfg in extract_cfg["cases"].items():
        print(f"\nExtracting production diagnostics for case: {case_name}")

        command = [
            sys.executable,
            script,
            "--pvd",
            case_cfg["pvd"],
            "--cell-id",
            str(case_cfg["cell_id"]),
            "--cache",
            case_cfg["cache"],
            "--diagnostics-cache",
            case_cfg["diagnostics_cache"],
        ]

        append_list(command, "--variables", case_cfg.get("variables"))
        append_if_present(command, "--shape", case_cfg, "shape")
        append_if_present(command, "--block-filter", case_cfg, "block_filter")
        append_if_present(command, "--max-time-days", case_cfg, "max_time_days")

        if case_cfg.get("force_reextract", False):
            command.append("--force-reextract")

        run_command(command, dry_run=dry_run)


def generate_figure(
    name: str,
    fig_cfg: dict[str, Any],
    pvbatch: str,
    *,
    dry_run: bool = False,
) -> None:
    """Generate a single figure based on its manifest entry."""
    fig_type = fig_cfg["type"]

    print("\n" + "=" * 80)
    print(f"Figure: {name}")
    print(f"Type:   {fig_type}")
    print("=" * 80)

    if fig_type == "paraview_layout_png":
        run_paraview_layout_png(name, fig_cfg, pvbatch, dry_run=dry_run)

    elif fig_type == "csv_then_matplotlib":
        run_paraview_csv_extract(fig_cfg["paraview_extract"], pvbatch, dry_run=dry_run)
        run_matplotlib_plot(fig_cfg["matplotlib_plot"], dry_run=dry_run)

    elif fig_type == "paraview_time_series_then_matplotlib":
        run_paraview_time_series(fig_cfg["paraview_render"], pvbatch, dry_run=dry_run)
        run_matplotlib_plot(fig_cfg["matplotlib_plot"], dry_run=dry_run)

    elif fig_type == "paraview_comparison_then_matplotlib":
        render_cfg = fig_cfg["paraview_render"]
        for panel_name, panel_cfg in render_cfg["panels"].items():
            print(f"\nRendering comparison panel: {panel_name}")
            panel_cfg = dict(panel_cfg)
            panel_cfg["script"] = render_cfg["script"]
            run_paraview_time_series(panel_cfg, pvbatch, dry_run=dry_run)

        run_matplotlib_plot(fig_cfg["matplotlib_plot"], dry_run=dry_run)

    elif fig_type == "pvd_timeseries_then_matplotlib":
        run_well_timeseries_extraction(fig_cfg["data_extract"], dry_run=dry_run)
        run_production_diagnostics(fig_cfg, dry_run=dry_run)
    elif fig_type == "benchmark_reference_then_matplotlib":
        run_benchmark_profile_extract(fig_cfg["porepy_extract"], dry_run=dry_run)
        run_benchmark_comparison_plot(fig_cfg["matplotlib_plot"], dry_run=dry_run)

    else:
        raise RuntimeError(f"Unknown figure type: {fig_type}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate geothermal-flow figures.")
    parser.add_argument(
        "--config",
        default="geothermal_flow/configs/figures.yaml",
        help="Path to figures.yaml.",
    )
    parser.add_argument(
        "--figures",
        nargs="*",
        default=None,
        help="Optional list of figure keys to generate. If omitted, generate all.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the figure-generation workflow."""
    args = parse_args()

    config = load_yaml(args.config)
    pvbatch = config.get("pvbatch", "pvbatch")
    figures = config["figures"]

    selected = args.figures or list(figures.keys())

    missing = [name for name in selected if name not in figures]
    if missing:
        raise KeyError(f"Unknown figure names: {missing}")

    for name in selected:
        generate_figure(
            name=name,
            fig_cfg=figures[name],
            pvbatch=pvbatch,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
