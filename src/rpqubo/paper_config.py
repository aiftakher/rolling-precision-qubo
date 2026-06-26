"""Centralized settings for reproducing the paper examples."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata


@dataclass(frozen=True)
class AnnealConfig:
    num_reads: int
    sweeps: int
    seed: int | None


@dataclass(frozen=True)
class PaperConfig:
    name: str
    anneal: AnnealConfig
    penalties: dict[str, float] = field(default_factory=dict)
    dynamic_penalty: bool = False
    rescale_target: float | None = None
    include_offset_in_rescale: bool = False
    cardinality_slack_encoding: str | None = None
    notes: str = ""


EXAMPLE1_BIT_GROWTH = PaperConfig(
    name="example1_bit_growth",
    anneal=AnnealConfig(num_reads=200, sweeps=2500, seed=11),
)

EXAMPLE1_ZOOM = PaperConfig(
    name="example1_zoom",
    anneal=AnnealConfig(num_reads=300, sweeps=4000, seed=11),
    dynamic_penalty=False,
    rescale_target=10.0,
    include_offset_in_rescale=True,
)

EXAMPLE2_BIT_GROWTH = PaperConfig(
    name="example2_bit_growth",
    anneal=AnnealConfig(num_reads=200, sweeps=2500, seed=13),
    penalties={"ineq": 100.0},
)

EXAMPLE2_ZOOM = PaperConfig(
    name="example2_zoom",
    anneal=AnnealConfig(num_reads=300, sweeps=4000, seed=13),
    penalties={"ineq": 100.0},
    dynamic_penalty=True,
    rescale_target=10.0,
    include_offset_in_rescale=True,
)

ALAN_BIT_GROWTH = PaperConfig(
    name="alan_bit_growth",
    anneal=AnnealConfig(num_reads=300, sweeps=3000, seed=7),
    penalties={"e1": 500.0, "e2": 500.0, "link": 500.0, "card": 500.0},
    cardinality_slack_encoding="sbe",
)

ALAN_TABLE6_REFERENCE = PaperConfig(
    name="paper_table6_reference",
    anneal=AnnealConfig(num_reads=300, sweeps=4000, seed=13),
    penalties={"e1": 200.0, "e2": 200.0, "link": 300.0, "card": 200.0},
    dynamic_penalty=True,
    rescale_target=10.0,
    include_offset_in_rescale=True,
    cardinality_slack_encoding="integer",
    notes="Legacy Table 6 settings may slightly differ.",
)

ALAN_MANUSCRIPT_UNIFORM_500 = PaperConfig(
    name="manuscript_uniform_500",
    anneal=AnnealConfig(num_reads=300, sweeps=4000, seed=13),
    penalties={"e1": 500.0, "e2": 500.0, "link": 500.0, "card": 500.0},
    dynamic_penalty=True,
    rescale_target=10.0,
    include_offset_in_rescale=True,
    cardinality_slack_encoding="integer",
)

ALAN_PENALTY_SENSITIVITY = PaperConfig(
    name="alan_penalty_sensitivity",
    anneal=AnnealConfig(num_reads=300, sweeps=4000, seed=13),
    dynamic_penalty=False,
    rescale_target=10.0,
    include_offset_in_rescale=True,
    cardinality_slack_encoding="integer",
)


def package_versions() -> dict[str, str]:
    """Return dependency versions included in generated reports."""

    versions: dict[str, str] = {}
    for package in (
        "rolling-precision-qubo",
        "dimod",
        "dwave-neal",
        "dwave-samplers",
        "numpy",
        "PyYAML",
        "pytest",
        "ruff",
        "mypy",
    ):
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            continue
    return versions
