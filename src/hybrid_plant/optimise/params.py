"""
optimise/params.py
──────────────────
Loads all model parameters from FullConfig + time-series data and assembles
them into an immutable ``OptParams`` dataclass.

Design §2.2 parameter inventory is implemented here.  Every value traces
directly to a YAML key or CSV column — nothing is hardcoded.

Precomputed scalar constants (§2.2):
  A_N        Σ_{t=1..N} df[t]                     annuity factor, project life
  A_n        Σ_{t=1..n} df[t]                     annuity factor, debt tenure
  emi_factor a = r(1+r)^n / ((1+r)^n − 1)         EMI factor
  phi        Φ = debt_frac·a·A_n + eq_frac·roe·A_N financing recovery factor
  G_*        Σ_t df[t]·(1+esc)^(t-1)              escalated-OPEX discount sums
  D_s/D_w/D_b Σ_t df[t]·d_[t]                    discounted-degradation sums

C-rate convention (D3):
  C-rates are treated as fixed user parameters.  ``solver.yaml`` exposes
  ``bess_charge_c_rate`` and ``bess_discharge_c_rate`` under
  ``decision_variables``.  The model reads ``fixed_value`` if present;
  otherwise falls back to ``max``.  Both values are currently ``max = 1.0``,
  consistent with all reference benchmark solutions in ``conftest.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hybrid_plant._paths import find_project_root
from hybrid_plant.config_loader import FullConfig
from hybrid_plant.constants import (
    CRORE_TO_RS,
    HOURS_PER_DAY,
    LAKH_TO_RS,
    MONTHS_PER_YEAR,
    MWH_TO_KWH,
    PERCENT_TO_DECIMAL,
)
from hybrid_plant.data_loader import operating_value
from hybrid_plant.energy.grid_interface import GridInterface
from hybrid_plant.energy.year1_engine import _build_hourly_discom_tariff


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_degrad_curve(path: Path, value_col: str) -> dict[int, float]:
    """
    Load a degradation CSV into a {year (int): value (float)} dict.

    The CSV must contain a ``year`` column and a column named ``value_col``
    (case-insensitive after stripping whitespace).
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    if "year" not in df.columns:
        raise ValueError(f"'year' column not found in {path}")
    col = value_col.lower()
    if col not in df.columns:
        raise ValueError(f"'{value_col}' column not found in {path}")
    return dict(zip(df["year"].astype(int), df[col]))


def _g_esc(df_arr: np.ndarray, esc: float) -> float:
    """
    Escalated-OPEX discount sum:  Σ_{t=1..N} df[t] × (1 + esc)^(t-1).

    ``df_arr`` is zero-indexed (df_arr[0] = df[1], df_arr[t-1] = df[t]).
    """
    n = len(df_arr)
    return float(np.dot(df_arr, (1.0 + esc) ** np.arange(n)))


# ─────────────────────────────────────────────────────────────────────────────
# Optional-constraint configuration (§3.6)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class OptionalConstraintsConfig:
    """
    Toggleable PPA-contract constraints, read from ``solver.yaml`` →
    ``solver.constraints``.  All percents are on a 0–100 scale (as written in
    the YAML).  Every toggle defaults to a no-op so the base model is unchanged
    when the block is absent or all constraints are disabled.

    Hard LP constraints (enforced in ``constraints/optional.py``):
      plant_cuf              min% ≤ Σ busbar_export / (P·n_steps)·100 ≤ max%
      minimum_bess_capacity  E_b (= nb·cs, Year-1 SOH) ≥ min_mwh
      minimum_bess_discharge Σ η_d·dis ≥ min_annual_mwh · n_years  (busbar MWh)
      re_penetration         min% ≤ Σ(load − ddraw) / Σ load ·100 ≤ max%

    Step-6b additions (Required + easy Nice):
      peak_supply            Σ_{peak}(load−ddraw) ≥ min% · Σ_{peak} load
      peak_bess_discharge    Σ_{peak} η_d·dis ≥ min_annual_mwh · n_years
      poi_capacity           sd+wd+η_d·dis ≤ poi_mw           (per hour)
      sanctioned_demand      ddraw ≤ demand_mw                (per hour)
      min_grid_drawal        Σ ddraw ≥ min_annual_mwh · n_years
      energy_purchase_cap    Σ(load−ddraw) ≤ max_annual_mwh · n_years
      land_area              S·acre_s + W·acre_w ≤ available_acres

    Report-only viability gate (NOT a hard LP row — checked post-solve so an
    LP row is never coupled to the horizon-scaled objective expression):
      minimum_savings_npv    savings_npv ≥ min_value

    ``*_hours`` tuples are 0-indexed (converted from the 1-indexed YAML).
    """

    plant_cuf_enabled:             bool  = False
    plant_cuf_min_pct:             float = 0.0
    plant_cuf_max_pct:             float = 100.0
    min_bess_capacity_enabled:     bool  = False
    min_bess_capacity_mwh:         float = 0.0
    min_bess_discharge_enabled:    bool  = False
    min_bess_discharge_annual_mwh: float = 0.0
    re_penetration_enabled:        bool  = False
    re_penetration_min_pct:        float = 0.0
    re_penetration_max_pct:        float = 100.0
    min_savings_npv_enabled:       bool  = True
    min_savings_npv_value:         float = 0.0

    # ── Step 6b: Required ─────────────────────────────────────────────────────
    peak_supply_enabled:           bool  = False
    peak_supply_min_pct:           float = 0.0
    peak_supply_hours:             tuple = ()      # 0-indexed hours-of-day
    peak_discharge_enabled:        bool  = False
    peak_discharge_annual_mwh:     float = 0.0
    peak_discharge_hours:          tuple = ()      # 0-indexed hours-of-day
    poi_enabled:                   bool  = False
    poi_mw:                        float = 0.0
    sanctioned_demand_enabled:     bool  = False
    sanctioned_demand_mw:          float = 0.0
    # ── Step 6b: easy Nice-to-have ────────────────────────────────────────────
    min_grid_drawal_enabled:       bool  = False
    min_grid_drawal_annual_mwh:    float = 0.0
    energy_purchase_cap_enabled:   bool  = False
    energy_purchase_cap_annual_mwh: float = 0.0
    land_area_enabled:             bool  = False
    land_available_acres:          float = 0.0
    land_solar_acre_per_mw:        float = 0.0
    land_wind_acre_per_mw:         float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# OptParams dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class OptParams:
    """
    Immutable bundle of all numeric parameters for the Pyomo model.

    Arrays are indexed 0-based internally:
      - Hourly arrays: length 8760, index h = 0 … 8759
      - Annual arrays: length project_life, index i = 0 → year 1

    Energy units: MWh.  Monetary units: INR (raw, unscaled).
    Scaling to Crore/GWh is applied in ``build.py``.
    """

    # ── Hourly time-series (shape: 8760) ──────────────────────────────────────
    cuf_s:  np.ndarray   # solar capacity factor [0, 1]
    cuf_w:  np.ndarray   # wind capacity factor  [0, 1]
    load:   np.ndarray   # client load (MWh per hour)
    tod:    np.ndarray   # blended DISCOM ToD tariff (INR/kWh)

    # ── Physics scalars ───────────────────────────────────────────────────────
    lf:     float        # grid loss factor  [0, 1]
    eta_c:  float        # BESS charge efficiency  [0, 1]
    eta_d:  float        # BESS discharge efficiency  [0, 1]
    cs:     float        # container size (MWh)
    aux_pc: float        # aux draw per active container per hour (MWh)
    crc:    float        # fixed charge C-rate (fraction of E_b per hour)
    crd:    float        # fixed discharge C-rate

    # ── Grid charges ──────────────────────────────────────────────────────────
    wheel:    float      # blended wheeling rate (INR/kWh)
    tax:      float      # blended electricity-tax rate (INR/kWh)
    cap_rate: float      # blended (CTU+STU+SLDC) capacity charge (INR/MW/month)

    # ── CAPEX rates ───────────────────────────────────────────────────────────
    solar_rate:  float   # INR per DC MWp
    ac_dc:       float   # AC-to-DC ratio (dimensionless)
    wind_rate:   float   # INR per MW
    bess_rate:   float   # INR per MWh
    trans_fixed: float   # fixed transmission CAPEX (INR)

    # ── OPEX base rates (Year-1 values, unescalated) ──────────────────────────
    solar_om_rate:     float   # INR per DC MWp per year
    solar_om_esc:      float   # escalation rate (decimal)
    wind_om_rate:      float   # INR per MW per year
    wind_om_esc:       float
    land_lease_monthly: float  # INR per month (base year)
    land_esc:           float
    bess_om_rate:       float  # INR per MWh per year (no escalation)
    solar_trans_om_rate: float # INR per DC MWp per year (no escalation)
    wind_trans_om_rate:  float # INR per MW per year (no escalation)
    insurance_pct:       float # fraction of total CAPEX per year (no escalation)

    # ── Financing ─────────────────────────────────────────────────────────────
    wacc:         float   # weighted-average cost of capital (decimal)
    debt_frac:    float   # debt fraction of CAPEX (decimal)
    eq_frac:      float   # equity fraction of CAPEX (decimal)
    r:            float   # debt interest rate (decimal)
    tenure:       int     # debt tenure (years)
    roe:          float   # equity return-on-equity rate (decimal)
    project_life: int     # project life (years)

    # ── Degradation operating-value series (shape: project_life) ─────────────
    # d_[i] is the operating value DURING year (i+1):
    #   d_[0] = 1.0 for all streams (Year 1, fresh)
    #   d_[i] = curve[i]  for i ≥ 1  (end-of-year i = start-of-year i+1)
    d_s: np.ndarray   # solar degradation
    d_w: np.ndarray   # wind degradation
    d_b: np.ndarray   # BESS SOH

    # ── Decision-variable bounds (from solver.yaml) ───────────────────────────
    s_min:  float
    s_max:  float
    w_min:  float
    w_max:  float
    p_min:  float
    p_max:  float
    nb_min: int
    nb_max: int

    # ── Precomputed scalar constants ──────────────────────────────────────────
    df:         np.ndarray  # discount factors df[i] = (1+wacc)^(-(i+1)), shape (N,)
    A_N:        float       # Σ_{t=1..N} df[t]  annuity factor, project life
    A_n:        float       # Σ_{t=1..n} df[t]  annuity factor, debt tenure
    emi_factor: float       # a = r(1+r)^n / ((1+r)^n − 1)
    phi:        float       # Φ = debt_frac·a·A_n + eq_frac·roe·A_N

    G_solar_om: float   # Σ_t df[t]·(1+solar_esc)^(t-1)
    G_wind_om:  float   # Σ_t df[t]·(1+wind_esc)^(t-1)
    G_land:     float   # Σ_t df[t]·(1+land_esc)^(t-1)
    # Non-escalating components share A_N — stored explicitly for clarity
    G_bess_om:       float  # = A_N
    G_solar_trans:   float  # = A_N
    G_wind_trans:    float  # = A_N
    G_insurance:     float  # = A_N

    D_s: float   # Σ_t df[t]·d_s[t]   discounted solar degradation sum
    D_w: float   # Σ_t df[t]·d_w[t]
    D_b: float   # Σ_t df[t]·d_b[t]

    # ── Optional-constraint parameters (§3.6) ─────────────────────────────────
    # Hard toggleable PPA-contract constraints (constraints/optional.py)
    opt_constraints: OptionalConstraintsConfig

    # RE penetration penalty (hourly mask + config values)
    re_pen_penalty_enabled: bool
    re_pen_min_pct:         float   # floor (decimal, e.g. 0.50)
    re_pen_penalty_hours:   tuple   # 0-indexed hours subject to penalty (empty = all)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_params(config: FullConfig, data: dict[str, Any]) -> OptParams:
    """
    Assemble an ``OptParams`` from a validated ``FullConfig`` and timeseries data.

    Parameters
    ----------
    config : FullConfig
        Loaded via ``config_loader.load_config()``.
    data : dict
        Loaded via ``data_loader.load_timeseries_data(config)``.
        Must contain ``solar_cuf``, ``wind_cuf``, ``load_profile``.

    Returns
    -------
    OptParams
        Frozen dataclass with all model parameters and precomputed constants.
    """
    root = find_project_root()

    # ── Time-series ───────────────────────────────────────────────────────────
    cuf_s  = data["solar_cuf"]         # already [0,1] from data_loader
    cuf_w  = data["wind_cuf"]
    load   = data["load_profile"]      # MWh per hour
    tod    = _build_hourly_discom_tariff(config, n_hours=len(load))  # INR/kWh

    # ── Physics ───────────────────────────────────────────────────────────────
    lf    = GridInterface(config).loss_factor

    bess  = config.bess["bess"]
    eta_c = float(bess["efficiency"]["charge_efficiency"])
    eta_d = float(bess["efficiency"]["discharge_efficiency"])
    cs    = float(bess["container"]["size_mwh"])
    aux_pc = float(bess["container"]["auxiliary_consumption_mwh_per_day"]) / HOURS_PER_DAY

    # C-rates: fixed per D3.  Read fixed_value if present, otherwise max.
    dv_cfg = config.solver["solver"]["decision_variables"]
    crc = float(
        dv_cfg.get("bess_charge_c_rate", {}).get("fixed_value")
        or dv_cfg.get("bess_charge_c_rate", {}).get("max", 1.0)
    )
    crd = float(
        dv_cfg.get("bess_discharge_c_rate", {}).get("fixed_value")
        or dv_cfg.get("bess_discharge_c_rate", {}).get("max", 1.0)
    )

    # ── Grid charges ─────────────────────────────────────────────────────────
    rc       = config.finance["regulatory_charges"]
    ht_frac  = config.regulatory["regulatory"]["connection"]["ht_lt_split_percent"] * PERCENT_TO_DECIMAL
    lt_frac  = 1.0 - ht_frac

    wheel = (
        ht_frac * float(rc["ht"]["wheeling_charge_inr_per_kwh"])
        + lt_frac * float(rc["lt"]["wheeling_charge_inr_per_kwh"])
    )
    tax = (
        ht_frac * float(rc["ht"]["electricity_tax_inr_per_kwh"])
        + lt_frac * float(rc["lt"]["electricity_tax_inr_per_kwh"])
    )
    cap_rate = (
        ht_frac * (
            float(rc["ht"]["ctu_charge_inr_per_mw_per_month"])
            + float(rc["ht"]["stu_charge_inr_per_mw_per_month"])
            + float(rc["ht"]["sldc_charge_inr_per_mw_per_month"])
        )
        + lt_frac * (
            float(rc["lt"]["ctu_charge_inr_per_mw_per_month"])
            + float(rc["lt"]["stu_charge_inr_per_mw_per_month"])
            + float(rc["lt"]["sldc_charge_inr_per_mw_per_month"])
        )
    )

    # ── CAPEX rates ───────────────────────────────────────────────────────────
    capex_cfg   = config.finance["capex"]
    solar_rate  = float(capex_cfg["solar"]["cost_per_mwp"])      # INR per DC MWp
    ac_dc       = float(capex_cfg["solar"]["ac_dc_ratio"])
    wind_rate   = float(capex_cfg["wind"]["cost_per_mw"])
    bess_rate   = float(capex_cfg["bess"]["cost_per_mwh"])
    trans_fixed = (
        float(capex_cfg["transmission"]["length_km"])
        * float(capex_cfg["transmission"]["cost_per_km"])
    )

    # ── OPEX base rates ───────────────────────────────────────────────────────
    opex_cfg = config.finance["opex"]

    solar_om_rate      = float(opex_cfg["solar"]["rate_lakh_per_mwp"]) * LAKH_TO_RS
    solar_om_esc       = float(opex_cfg["solar"]["escalation_percent"]) * PERCENT_TO_DECIMAL
    wind_om_rate       = float(opex_cfg["wind"]["rate_lakh_per_mw"]) * LAKH_TO_RS
    wind_om_esc        = float(opex_cfg["wind"]["escalation_percent"]) * PERCENT_TO_DECIMAL
    land_lease_monthly = (
        float(opex_cfg["land_lease"]["base_monthly_cost_crore"]) * CRORE_TO_RS
    )
    land_esc           = float(opex_cfg["land_lease"]["escalation_percent"]) * PERCENT_TO_DECIMAL
    bess_om_rate       = float(opex_cfg["bess"]["rate_lakh_per_mwh"]) * LAKH_TO_RS
    solar_trans_om_rate = (
        float(opex_cfg["solar_transmission"]["rate_lakh_per_mwp"]) * LAKH_TO_RS
    )
    wind_trans_om_rate  = (
        float(opex_cfg["wind_transmission"]["rate_lakh_per_mw"]) * LAKH_TO_RS
    )
    insurance_pct = (
        float(opex_cfg["insurance"]["percent_of_total_capex"]) * PERCENT_TO_DECIMAL
    )

    # ── Financing ─────────────────────────────────────────────────────────────
    fin          = config.finance["financing"]
    project_life = int(config.project["project"]["project_life_years"])
    debt_frac    = float(fin["debt_percent"])   * PERCENT_TO_DECIMAL
    eq_frac      = float(fin["equity_percent"]) * PERCENT_TO_DECIMAL
    r            = float(fin["debt"]["interest_rate_percent"]) * PERCENT_TO_DECIMAL
    tenure       = int(fin["debt"]["tenure_years"])
    roe          = float(fin["equity"]["return_on_equity_percent"]) * PERCENT_TO_DECIMAL
    tax_rate     = float(fin["corporate_tax_rate_percent"]) * PERCENT_TO_DECIMAL

    wacc = debt_frac * r * (1.0 - tax_rate) + eq_frac * roe

    # ── Degradation curves ────────────────────────────────────────────────────
    gen_cfg = config.project["generation"]

    solar_eff_curve = _load_degrad_curve(
        root / gen_cfg["solar"]["degradation"]["file"], "efficiency"
    )
    wind_eff_curve = _load_degrad_curve(
        root / gen_cfg["wind"]["degradation"]["file"], "efficiency"
    )
    bess_soh_curve = _load_degrad_curve(
        root / config.bess["bess"]["degradation"]["file"], "soh"
    )

    d_s = np.array([operating_value(solar_eff_curve, y) for y in range(1, project_life + 1)])
    d_w = np.array([operating_value(wind_eff_curve,  y) for y in range(1, project_life + 1)])
    d_b = np.array([operating_value(bess_soh_curve,  y) for y in range(1, project_life + 1)])

    # ── Decision-variable bounds ──────────────────────────────────────────────
    dv = config.solver["solver"]["decision_variables"]
    s_min  = float(dv["solar_capacity_mw"]["min"])
    s_max  = float(dv["solar_capacity_mw"]["max"])
    w_min  = float(dv["wind_capacity_mw"]["min"])
    w_max  = float(dv["wind_capacity_mw"]["max"])
    p_min  = float(dv["ppa_capacity_mw"]["min"])
    p_max  = float(dv["ppa_capacity_mw"]["max"])
    nb_min = int(dv["bess_containers"]["min"])
    nb_max = int(dv["bess_containers"]["max"])

    # ── Precomputed scalar constants ──────────────────────────────────────────
    t_arr = np.arange(1, project_life + 1)
    df_arr = (1.0 + wacc) ** (-t_arr.astype(float))   # df[i] = (1+w)^{-(i+1)}

    A_N = float(np.sum(df_arr))
    A_n = float(np.sum(df_arr[:tenure]))

    if r > 0.0:
        emi_factor = r * (1.0 + r) ** tenure / ((1.0 + r) ** tenure - 1.0)
    else:
        emi_factor = 1.0 / tenure

    phi = debt_frac * emi_factor * A_n + eq_frac * roe * A_N

    G_solar_om    = _g_esc(df_arr, solar_om_esc)
    G_wind_om     = _g_esc(df_arr, wind_om_esc)
    G_land        = _g_esc(df_arr, land_esc)
    G_bess_om     = A_N
    G_solar_trans = A_N
    G_wind_trans  = A_N
    G_insurance   = A_N

    D_s = float(np.dot(df_arr, d_s))
    D_w = float(np.dot(df_arr, d_w))
    D_b = float(np.dot(df_arr, d_b))

    # ── Optional-constraint parameters ────────────────────────────────────────
    cons = config.solver["solver"].get("constraints", {})

    def _con(name: str) -> dict:
        return cons.get(name, {}) or {}

    pcuf  = _con("plant_cuf")
    mbc   = _con("minimum_bess_capacity")
    mbd   = _con("minimum_bess_discharge")
    repen = _con("re_penetration")
    msav  = _con("minimum_savings_npv")
    psup  = _con("peak_supply_obligation")
    pdis  = _con("peak_bess_discharge")
    poi   = _con("poi_capacity")
    sdem  = _con("sanctioned_demand")
    mgd   = _con("min_grid_drawal")
    epc   = _con("energy_purchase_cap")
    land  = _con("land_area")

    def _hours0(cfg: dict) -> tuple:
        """1-indexed hours-of-day from YAML → 0-indexed tuple."""
        return tuple(int(h) - 1 for h in cfg.get("peak_hours", []))

    opt_constraints = OptionalConstraintsConfig(
        plant_cuf_enabled=bool(pcuf.get("enabled", False)),
        plant_cuf_min_pct=float(pcuf.get("min_percent", 0.0)),
        plant_cuf_max_pct=float(pcuf.get("max_percent", 100.0)),
        min_bess_capacity_enabled=bool(mbc.get("enabled", False)),
        min_bess_capacity_mwh=float(mbc.get("min_mwh", 0.0)),
        min_bess_discharge_enabled=bool(mbd.get("enabled", False)),
        min_bess_discharge_annual_mwh=float(mbd.get("min_annual_mwh", 0.0)),
        re_penetration_enabled=bool(repen.get("enabled", False)),
        re_penetration_min_pct=float(repen.get("min_percent", 0.0)),
        re_penetration_max_pct=float(repen.get("max_percent", 100.0)),
        min_savings_npv_enabled=bool(msav.get("enabled", True)),
        min_savings_npv_value=float(msav.get("min_value", 0.0)),
        # Step 6b — Required
        peak_supply_enabled=bool(psup.get("enabled", False)),
        peak_supply_min_pct=float(psup.get("min_percent", 0.0)),
        peak_supply_hours=_hours0(psup),
        peak_discharge_enabled=bool(pdis.get("enabled", False)),
        peak_discharge_annual_mwh=float(pdis.get("min_annual_mwh", 0.0)),
        peak_discharge_hours=_hours0(pdis),
        poi_enabled=bool(poi.get("enabled", False)),
        poi_mw=float(poi.get("poi_mw", 0.0)),
        sanctioned_demand_enabled=bool(sdem.get("enabled", False)),
        sanctioned_demand_mw=float(sdem.get("demand_mw", 0.0)),
        # Step 6b — easy Nice-to-have
        min_grid_drawal_enabled=bool(mgd.get("enabled", False)),
        min_grid_drawal_annual_mwh=float(mgd.get("min_annual_mwh", 0.0)),
        energy_purchase_cap_enabled=bool(epc.get("enabled", False)),
        energy_purchase_cap_annual_mwh=float(epc.get("max_annual_mwh", 0.0)),
        land_area_enabled=bool(land.get("enabled", False)),
        land_available_acres=float(land.get("available_acres", 0.0)),
        land_solar_acre_per_mw=float(land.get("solar_acre_per_mw", 0.0)),
        land_wind_acre_per_mw=float(land.get("wind_acre_per_mw", 0.0)),
    )

    hrep = cons.get("hourly_re_penetration_penalty", {}) or {}
    re_pen_penalty_enabled = bool(hrep.get("enabled", False))
    re_pen_min_pct         = float(hrep.get("min_percent", 0.0)) * PERCENT_TO_DECIMAL
    # Convert 1-indexed penalty_hours (YAML) to 0-indexed
    _ph = hrep.get("penalty_hours", [])
    re_pen_penalty_hours = tuple(int(h) - 1 for h in _ph)

    return OptParams(
        # Time-series
        cuf_s=cuf_s,
        cuf_w=cuf_w,
        load=load,
        tod=tod,
        # Physics
        lf=lf,
        eta_c=eta_c,
        eta_d=eta_d,
        cs=cs,
        aux_pc=aux_pc,
        crc=crc,
        crd=crd,
        # Grid charges
        wheel=wheel,
        tax=tax,
        cap_rate=cap_rate,
        # CAPEX
        solar_rate=solar_rate,
        ac_dc=ac_dc,
        wind_rate=wind_rate,
        bess_rate=bess_rate,
        trans_fixed=trans_fixed,
        # OPEX
        solar_om_rate=solar_om_rate,
        solar_om_esc=solar_om_esc,
        wind_om_rate=wind_om_rate,
        wind_om_esc=wind_om_esc,
        land_lease_monthly=land_lease_monthly,
        land_esc=land_esc,
        bess_om_rate=bess_om_rate,
        solar_trans_om_rate=solar_trans_om_rate,
        wind_trans_om_rate=wind_trans_om_rate,
        insurance_pct=insurance_pct,
        # Financing
        wacc=wacc,
        debt_frac=debt_frac,
        eq_frac=eq_frac,
        r=r,
        tenure=tenure,
        roe=roe,
        project_life=project_life,
        # Degradation
        d_s=d_s,
        d_w=d_w,
        d_b=d_b,
        # Bounds
        s_min=s_min,
        s_max=s_max,
        w_min=w_min,
        w_max=w_max,
        p_min=p_min,
        p_max=p_max,
        nb_min=nb_min,
        nb_max=nb_max,
        # Precomputed constants
        df=df_arr,
        A_N=A_N,
        A_n=A_n,
        emi_factor=emi_factor,
        phi=phi,
        G_solar_om=G_solar_om,
        G_wind_om=G_wind_om,
        G_land=G_land,
        G_bess_om=G_bess_om,
        G_solar_trans=G_solar_trans,
        G_wind_trans=G_wind_trans,
        G_insurance=G_insurance,
        D_s=D_s,
        D_w=D_w,
        D_b=D_b,
        # Optional constraints
        opt_constraints=opt_constraints,
        re_pen_penalty_enabled=re_pen_penalty_enabled,
        re_pen_min_pct=re_pen_min_pct,
        re_pen_penalty_hours=re_pen_penalty_hours,
    )
