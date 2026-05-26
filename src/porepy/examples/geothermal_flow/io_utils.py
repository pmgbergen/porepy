"""YAML configuration utilities for the unified geothermal examples.

The configuration is intentionally two-layered:

* ``configs/defaults.yaml`` contains values common to all examples.
* ``configs/example*.yaml`` contains only values that distinguish that example.

The driver deep-merges both files before creating the PorePy model.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PACKAGE_DIR / "configs" / "defaults.yaml"


class ConfigurationError(ValueError):
    """Raised when a YAML configuration is missing required data."""


def as_float(value: Any) -> float:
    """Convert YAML scalar values such as ``1.0e-3`` into ``float``."""
    return float(value)


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``base`` recursively updated by ``override`` without mutation."""
    merged = deepcopy(dict(base))
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    with path.open("r", encoding="utf8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise ConfigurationError(f"YAML must contain a mapping at top level: {path}")
    return data


def load_config(path: str | Path, defaults_path: str | Path | None = None) -> dict[str, Any]:
    defaults_file = Path(defaults_path).expanduser().resolve() if defaults_path else DEFAULT_CONFIG
    config_file = Path(path).expanduser().resolve()
    config = deep_merge(load_yaml(defaults_file), load_yaml(config_file))
    config["_config_file"] = str(config_file)
    config["_defaults_file"] = str(defaults_file)
    validate_config(config)
    return config


def validate_config_old(config: Mapping[str, Any]) -> None:
    required = [
        "case_name", "geometry", "material", "well", "physics", "vtk",
        "time", "solver", "visualization",
    ]
    missing = [key for key in required if key not in config]
    if missing:
        raise ConfigurationError(f"Missing required config keys: {missing}")

    if config["geometry"] not in {"disconnected", "connected"}:
        raise ConfigurationError("geometry must be either 'disconnected' or 'connected'")

    for key in ["clogging_exponent", "reference_aperture", "injection_fraction"]:
        if key not in config["physics"]:
            raise ConfigurationError(f"physics.{key} must be specified by the example YAML")

    for key in ["end", "dt_init", "dt_min", "dt_max", "iter_optimal", "iter_relax"]:
        if key not in config["time"]:
            raise ConfigurationError(f"time.{key} must be specified by the example YAML")

    for key in ["end", "dt_init", "dt_min", "dt_max"]:
        if as_float(config["time"][key]) <= 0:
            raise ConfigurationError(f"time.{key} must be positive")

    if as_float(config["physics"]["reference_aperture"]) <= 0:
        raise ConfigurationError("physics.reference_aperture must be positive")
    if as_float(config["physics"]["injection_fraction"]) <= 0:
        raise ConfigurationError("physics.injection_fraction must be positive")

    for key in ["phz_file", "ptz_file", "directory"]:
        if not config["vtk"].get(key):
            raise ConfigurationError(f"vtk.{key} must be provided")


def is_benchmark_config(config: dict) -> bool:
    """Return True if the YAML config describes the 1D benchmark case."""
    return config.get("model", {}).get("kind") == "benchmark_1d"


def validate_config(config: dict) -> None:
    """Validate a geothermal-flow simulation configuration."""

    required_common = [
        "case_name",
        "geometry",
        "time",
        "material",
        "solver",
        "vtk",
        "visualization",
    ]

    for key in required_common:
        if key not in config:
            raise ConfigurationError(f"Missing required configuration key: {key}")

    if is_benchmark_config(config):
        validate_benchmark_config(config)
    else:
        validate_example_config(config)


def validate_example_config(config: dict) -> None:
    """Validate Example 1--3 fractured-reservoir configuration."""

    if config["geometry"] not in {"disconnected", "connected"}:
        raise ConfigurationError("geometry must be either 'disconnected' or 'connected'")

    required_example = [
        "physics",
        "well",
    ]

    for key in required_example:
        if key not in config:
            raise ConfigurationError(f"Missing required example configuration key: {key}")

    required_material = [
        "residual_aperture",
        "permeability",
        "normal_permeability",
        "fracture_permeability",
        "porosity",
        "thermal_conductivity",
        "density",
        "specific_heat_capacity",
    ]

    for key in required_material:
        if key not in config["material"]:
            raise ConfigurationError(f"Missing material parameter: {key}")


def validate_benchmark_config(config: dict) -> None:
    """Validate the 1D benchmark configuration."""

    if config["geometry"] != "benchmark_horizontal":
        raise ConfigurationError(
            "benchmark geometry must be 'benchmark_horizontal'"
        )

    required_benchmark = [
        "boundary_conditions",
        "initial_conditions",
        "relative_permeability",
    ]

    for key in required_benchmark:
        if key not in config:
            raise ConfigurationError(f"Missing required benchmark configuration key: {key}")

    required_material = [
        "permeability",
        "porosity",
        "thermal_conductivity",
        "density",
        "specific_heat_capacity",
    ]

    for key in required_material:
        if key not in config["material"]:
            raise ConfigurationError(f"Missing benchmark material parameter: {key}")

    bc = config["boundary_conditions"]

    for section in ["pressure", "temperature", "z_nacl"]:
        if section not in bc:
            raise ConfigurationError(
                f"Missing benchmark boundary_conditions section: {section}"
            )

    ic = config["initial_conditions"]

    for section in ["pressure", "temperature", "z_nacl"]:
        if section not in ic:
            raise ConfigurationError(
                f"Missing benchmark initial_conditions section: {section}"
            )


def write_yaml(data: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as stream:
        yaml.safe_dump(dict(data), stream, sort_keys=False)
