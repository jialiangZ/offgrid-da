"""Variable registry and cross-dataset variable mapping.

All names below were verified against the real zarr / parquet schemas in
Phase 0 (see docs/DATA_AUDIT.md and docs/VARIABLE_MAPPING.md):

- Pangu zarr data variables (11): 2m_temperature, 10m_u_component_of_wind, ...
- ERA5 zarr data variables (62): 2m_temperature (units K), ...
- METAR parquet columns (43): temperature (K), dewpoint (K), windDir (deg),
  windSpeed (m/s), seaLevelPress (Pa), ... plus ``<var>DD`` descriptor flags.

Machine variable IDs are stable integers used by observation tokens.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class VariableSpec:
    """A physical quantity in the model's internal variable space."""

    variable_id: int
    name: str                  # internal canonical name (matches grid datasets)
    units: str                 # canonical units after any conversion
    metar_column: str | None   # METAR parquet column (None => not observed at surface)
    metar_dd_column: str | None  # METAR data-descriptor (DD) flag column
    valid_range: tuple[float, float]  # physical sanity range in canonical units
    default_obs_error: float   # assumed obs error std (to be calibrated later)


@dataclass(frozen=True)
class VariableRegistry:
    variables: dict[str, VariableSpec] = field(default_factory=dict)

    def register(self, spec: VariableSpec) -> None:
        if spec.name in self.variables:
            raise ValueError(f"duplicate variable: {spec.name}")
        self.variables[spec.name] = spec

    def by_id(self, variable_id: int) -> VariableSpec:
        for spec in self.variables.values():
            if spec.variable_id == variable_id:
                return spec
        raise KeyError(variable_id)

    def get(self, name: str) -> VariableSpec:
        return self.variables[name]


def build_default_registry() -> VariableRegistry:
    """Default registry: surface variables available in all three sources.

    Ordering matches the plan (T2M first, then U10/V10 for the multi-variable
    stage). METAR windDir/windSpeed must be converted to u/v with
    ``u = -V sin(theta)``, ``v = -V cos(theta)`` (meteorological "from"
    convention, MADIS DD = degrees from north).
    """
    reg = VariableRegistry()
    reg.register(
        VariableSpec(
            variable_id=0,
            name="2m_temperature",
            units="K",
            metar_column="temperature",
            metar_dd_column="temperatureDD",
            valid_range=(180.0, 340.0),
            default_obs_error=1.0,
        )
    )
    reg.register(
        VariableSpec(
            variable_id=1,
            name="10m_u_component_of_wind",
            units="m s-1",
            metar_column=None,  # derived from windDir/windSpeed in later stage
            metar_dd_column=None,
            valid_range=(-60.0, 60.0),
            default_obs_error=1.5,
        )
    )
    reg.register(
        VariableSpec(
            variable_id=2,
            name="10m_v_component_of_wind",
            units="m s-1",
            metar_column=None,
            metar_dd_column=None,
            valid_range=(-60.0, 60.0),
            default_obs_error=1.5,
        )
    )
    return reg


DEFAULT_REGISTRY = build_default_registry()

# Grid-dataset name mapping. Verified: Pangu and ERA5 zarr use identical
# variable names for their 11 common variables.
GRID_VARIABLE_MAP = {
    "2m_temperature": {"pangu": "2m_temperature", "era5": "2m_temperature"},
    "10m_u_component_of_wind": {"pangu": "10m_u_component_of_wind", "era5": "10m_u_component_of_wind"},
    "10m_v_component_of_wind": {"pangu": "10m_v_component_of_wind", "era5": "10m_v_component_of_wind"},
}

# METAR DD data-descriptor flags — semantics determined from official MADIS
# docs (S/T/V/C/Z) plus empirical characterization over Jan-2020 partitions
# (B/G/Q/X, see docs/DATA_AUDIT.md §4):
#   Z = missing (0% values) — the only flag with no value
#   all other flags carry plausible values; V dominates (65% of rows, mostly
#   AUTO stations), C = unit-converted (e.g. visibility miles->m)
# The WB2 parquet export carries ONLY these DD flags; the numeric MADIS
# quality codes (QC levels 0-3) are not part of the export.
DD_FLAG_MEANING = {
    "S": "single_value",
    "T": "time_averaged",
    "V": "variable_value_dominant_on_auto_stations",
    "C": "unit_converted",
    "B": "both_groups_rare",
    "G": "rare_flag",
    "Q": "derived_or_quality_substituted_rare",
    "X": "extreme_outlier_rare",
    "Z": "missing",
}

# Numeric quality code used in observation tokens. Higher = better.
# Empirical basis: only Z lacks values; X rows show implausible constants
# (e.g. 333.15 K = 60 C) -> excluded by default QC.
DD_QUALITY_CODE = {
    "S": 3,   # plain single report
    "C": 3,   # unit-converted value (e.g. visibility)
    "V": 2,   # auto-station dominant flag; values verified plausible
    "T": 2,   # time-averaged
    "B": 2,   # rare; values present
    "G": 2,   # rare; values present
    "Q": 1,   # rare; values present, treat with lower trust
    "X": 0,   # implausible extremes observed
    "Z": 0,   # missing
}
