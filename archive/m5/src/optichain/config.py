"""Load and validate config.yaml."""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class DataConfig(BaseModel):
    raw_dir: Path
    processed_dir: Path
    sales_file: str = "sales_train_evaluation.csv"
    store_id: str
    max_day: int = Field(gt=0, le=1941)


class SeriesConfig(BaseModel):
    items: list[str] = []


class ForecastConfig(BaseModel):
    horizon: int = 28
    fit_window_days: int = 730
    origins: list[int]
    models: list[str]
    sarima_grid: dict[str, list[int] | int]
    interval_levels: list[float]


class DiagnosticsConfig(BaseModel):
    ljung_box_lags: list[int]
    arch_lm_lag: int
    alpha: float = 0.05


class SimulationAssumptions(BaseModel):
    lead_time_days: int
    review_days: int
    cover_days: int
    target_csl: float = Field(gt=0, lt=1)
    holding_rate_annual: float
    shortage_penalty_rate: float
    order_cost: float
    case_pack: int = 1
    sensitivity: dict[str, list[float]]


class SimulationConfig(BaseModel):
    """How the simulation is run (method choices, not hypothetical world assumptions)."""
    forecast_model: str = "auto"  # "auto" = rolling selection on completed folds, or a model name
    default_model: str = "snaive"  # used by "auto" before any fold has completed
    baseline_window: int = 28
    warmup_days: int = 7
    include_fixed_models: bool = True


class Config(BaseModel):
    data: DataConfig
    series: SeriesConfig
    forecast: ForecastConfig
    diagnostics: DiagnosticsConfig
    simulation_assumptions: SimulationAssumptions
    simulation: SimulationConfig = SimulationConfig()
    seed: int = 42


def load_config(path: Path | str | None = None) -> Config:
    path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    cfg = Config.model_validate(yaml.safe_load(path.read_text()))
    # Relative data paths are resolved against the project root, not the cwd.
    for attr in ("raw_dir", "processed_dir"):
        p = getattr(cfg.data, attr)
        if not p.is_absolute():
            setattr(cfg.data, attr, PROJECT_ROOT / p)
    if max(cfg.forecast.origins) + cfg.forecast.horizon > cfg.data.max_day:
        raise ValueError("Last forecast origin + horizon exceeds data.max_day")
    return cfg
