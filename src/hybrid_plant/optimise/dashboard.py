"""
optimise/dashboard.py
─────────────────────
Self-contained HTML dashboards for a Pyomo optimization run.

Two outputs, both single files with inline CSS and base64-embedded charts (no
external assets — open in any browser, shareable, printable):

  render_detailed_dashboard(...)   — dense engineering board: KPI band, sizing,
                                     capital/financing, energy balance, tariff
                                     build-up, 25-yr table, OPEX, dispatch, and
                                     the ToD/flat/RTC reconciliation.
  render_executive_dashboard(...)  — one page: big KPI cards + two headline
                                     charts + a one-line verdict.

Headline economics are ToD-aware (optimal dispatch valued at hourly ToD =
the LP objective).  The flat FinanceEngine estimate and the RTC heuristic
oracle are shown as context.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

from hybrid_plant.config_loader import FullConfig  # noqa: E402
from hybrid_plant.constants import CRORE_TO_RS      # noqa: E402

# Palette
_TEAL = "#0f9d8f"
_TEAL_D = "#0b7a70"
_AMBER = "#f4a832"
_BLUE = "#4c9be8"
_GREEN = "#66bb6a"
_RED = "#ef5350"
_PURPLE = "#9b59b6"
_INK = "#1a2b32"


# ─────────────────────────────────────────────────────────────────────────────
# Metric extraction
# ─────────────────────────────────────────────────────────────────────────────

def _cr(v: float) -> float:
    return float(v) / CRORE_TO_RS


def _payback_year(annual_savings: np.ndarray) -> int | None:
    cum = 0.0
    for i, s in enumerate(annual_savings, start=1):
        cum += s
        if cum > 0:
            return i
    return None


def compute_metrics(result: dict[str, Any], config: FullConfig) -> dict[str, Any]:
    """Flatten the pipeline result into display-ready scalar metrics."""
    fi = result["finance"]
    sizing = result["sizing"]
    sb = fi["savings_breakdown"]
    ep = fi["energy_projection"]
    cap = fi["capex"]

    tod_annual = np.asarray(result["tod_annual_savings"])
    tod_npv = result["tod_savings_npv"]
    baseline = sb["baseline_annual_cost"]

    annual_load_mwh = sb["annual_load_kwh"] / 1000.0
    meter_y1  = float(ep["delivered_meter_mwh"][0])
    busbar_y1 = float(ep["delivered_pre_mwh"][0])
    P = sizing["P"]
    e_cap = float(result["year1"]["energy_capacity_mwh"])

    return {
        "project_name":   config.project["project"].get("name", "Hybrid RE Plant"),
        "location":       config.project["project"].get("location", ""),
        # sizing
        "S": sizing["S"], "W": sizing["W"], "P": P, "nb": sizing["nb"],
        "bess_mwh": e_cap,
        "charge_source": result["best_params"]["bess_charge_source"],
        # headline economics (ToD-aware)
        "tod_npv_cr":     _cr(tod_npv),
        "savings_pct_y1": tod_annual[0] / baseline * 100.0,
        "cum_savings_cr": _cr(float(np.sum(tod_annual))),
        "payback":        _payback_year(tod_annual),
        # context economics
        "flat_npv_cr":    _cr(fi["savings_npv"]),
        "rtc_npv_cr":     _cr(result["oracle_rtc_npv"]),
        "lp_obj_cr":      _cr(result["lp_objective_npv"]),
        # tariff / technical
        "lcoe":           fi["lcoe_inr_per_kwh"],
        "landed_y1":      fi["landed_tariff_series"][0],
        "landed_y25":     fi["landed_tariff_series"][-1],
        "discom_tariff":  sb["discom_tariff"],
        "re_penetration": meter_y1 / annual_load_mwh * 100.0,
        "plant_cuf":      busbar_y1 / (P * 8760.0) * 100.0 if P > 0 else 0.0,
        "total_capex_cr": _cr(cap["total_capex"]),
        "wacc":           fi["wacc"] * 100.0,
        # verify
        "verify_ok":      result["verify"].ok,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Charts → base64 data URIs
# ─────────────────────────────────────────────────────────────────────────────

def _fig_to_uri(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _chart_savings(result: dict, m: dict) -> str:
    years = np.arange(1, len(result["tod_annual_savings"]) + 1)
    ann = np.asarray(result["tod_annual_savings"]) / CRORE_TO_RS
    cum = np.cumsum(ann)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.bar(years, ann, color=_TEAL, alpha=0.85, label="Annual savings")
    axr = ax.twinx()
    axr.plot(years, cum, color=_PURPLE, lw=2, marker="o", ms=2.5, label="Cumulative")
    ax.set_xlabel("Year"); ax.set_ylabel("Annual (₹ Cr)")
    axr.set_ylabel("Cumulative (₹ Cr)", color=_PURPLE)
    ax.set_title("Client Savings — ToD-aware (₹ Cr)", fontweight="bold", fontsize=11)
    ax.grid(True, alpha=0.2)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axr.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
    return _fig_to_uri(fig)


def _chart_tariff(result: dict, m: dict) -> str:
    fi = result["finance"]
    lts = np.asarray(fi["landed_tariff_series"])
    years = np.arange(1, len(lts) + 1)
    disc = m["discom_tariff"]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(years, lts, color=_BLUE, lw=2, label="Landed tariff")
    ax.axhline(disc, color=_RED, lw=1.5, ls="--", label=f"DISCOM avg ₹{disc:.2f}")
    ax.fill_between(years, lts, disc, where=(lts < disc), alpha=0.18, color=_TEAL,
                    label="Savings band")
    ax.set_xlabel("Year"); ax.set_ylabel("₹ / kWh")
    ax.set_title("Landed Tariff vs DISCOM", fontweight="bold", fontsize=11)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
    return _fig_to_uri(fig)


def _chart_energy_mix(result: dict) -> str:
    ep = result["finance"]["energy_projection"]
    sb = result["finance"]["savings_breakdown"]
    years = np.arange(1, len(ep["delivered_meter_mwh"]) + 1)
    solar = np.asarray(ep["solar_direct_mwh"]) / 1e3
    wind  = np.asarray(ep["wind_direct_mwh"]) / 1e3
    bess  = np.asarray(ep["battery_mwh"]) / 1e3
    load  = sb["annual_load_kwh"] / 1000.0 / 1e3
    meter = np.asarray(ep["delivered_meter_mwh"]) / 1e3
    discom = np.maximum(load - meter, 0)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.stackplot(years, solar, wind, bess, discom,
                 labels=["Solar", "Wind", "BESS", "DISCOM draw"],
                 colors=[_AMBER, _BLUE, _GREEN, _RED], alpha=0.85)
    ax.set_xlabel("Year"); ax.set_ylabel("Energy (GWh)")
    ax.set_title("Energy Mix over 25 Years", fontweight="bold", fontsize=11)
    ax.legend(fontsize=8, loc="lower left"); ax.grid(True, alpha=0.2)
    return _fig_to_uri(fig)


def _chart_opex(result: dict) -> str:
    ob = result["finance"]["opex_breakdown"]
    years = np.arange(1, len(ob) + 1)
    keys = [("Solar O&M", "solar_om"), ("Wind O&M", "wind_om"), ("BESS O&M", "bess_om"),
            ("Transmission", None), ("Land", "land_lease"), ("Insurance", "insurance")]
    cols = [_AMBER, _BLUE, _GREEN, _PURPLE, "#ff7043", "#78909c"]
    series = []
    for label, key in keys:
        if key is None:
            series.append(np.array([x["solar_transmission_om"] + x["wind_transmission_om"]
                                    for x in ob]) / CRORE_TO_RS)
        else:
            series.append(np.array([x[key] for x in ob]) / CRORE_TO_RS)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.stackplot(years, *series, labels=[k[0] for k in keys], colors=cols, alpha=0.85)
    ax.set_xlabel("Year"); ax.set_ylabel("OPEX (₹ Cr)")
    ax.set_title("OPEX Stack over 25 Years", fontweight="bold", fontsize=11)
    ax.legend(fontsize=8, loc="upper left"); ax.grid(True, alpha=0.2)
    return _fig_to_uri(fig)


def _chart_dispatch_day(result: dict, params_load: np.ndarray, day: int = 250) -> str:
    d = result["lp_dispatch"]
    s = day * 24
    x = np.arange(24)
    sd = d["sd"][s:s+24]; wd = d["wd"][s:s+24]
    dis = d["dis"][s:s+24]; chg = d["chg"][s:s+24]; soc = d["soc"][s:s+24]
    load = params_load[s:s+24]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.4, 4.6), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1.4]})
    ax1.bar(x, sd, color=_AMBER, label="Solar→load")
    ax1.bar(x, wd, bottom=sd, color=_BLUE, label="Wind→load")
    ax1.bar(x, dis, bottom=sd + wd, color=_GREEN, label="BESS discharge")
    ax1.plot(x, load, color=_INK, lw=1.5, ls=":", label="Load")
    ax1.set_ylabel("MWh/h"); ax1.legend(fontsize=7, ncol=2)
    ax1.set_title(f"LP-optimal dispatch — Day {day} (busbar)", fontweight="bold", fontsize=11)
    ax1.grid(True, alpha=0.2)
    ax2.bar(x, chg, color=_GREEN, alpha=0.6, label="Charge")
    ax2.bar(x, -dis, color=_BLUE, alpha=0.7, label="Discharge")
    ax2r = ax2.twinx()
    ax2r.plot(x, soc, color=_PURPLE, lw=2, label="SOC")
    ax2.set_xlabel("Hour"); ax2.set_ylabel("MWh/h"); ax2r.set_ylabel("SOC (MWh)", color=_PURPLE)
    ax2.legend(fontsize=7, loc="upper left"); ax2.grid(True, alpha=0.2)
    return _fig_to_uri(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Shared CSS
# ─────────────────────────────────────────────────────────────────────────────

_CSS = """
:root{--bg:#f6f8f9;--card:#fff;--ink:#1a2b32;--muted:#5c7079;--line:#e2e8ea;
--teal:#0f9d8f;--teal-d:#0b7a70;--amber:#f4a832;--red:#ef5350;--good:#0f9d8f;}
@media(prefers-color-scheme:dark){:root{--bg:#0f1719;--card:#16232733;--ink:#e6eef0;
--muted:#9fb3bb;--line:#26383e;}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
line-height:1.45}
.wrap{max-width:1180px;margin:0 auto;padding:28px 22px 60px}
h1{font-size:26px;margin:0 0 2px}h2{font-size:16px;letter-spacing:.04em;text-transform:uppercase;
color:var(--muted);margin:34px 0 12px;border-bottom:1px solid var(--line);padding-bottom:6px}
.sub{color:var(--muted);font-size:14px;margin-bottom:8px}
.kpis{display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));margin:18px 0}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px 18px}
.kpi .lbl{font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}
.kpi .val{font-size:30px;font-weight:700;margin-top:4px}
.kpi .val.big{font-size:38px;color:var(--teal)}
.kpi .note{font-size:12.5px;color:var(--muted);margin-top:3px}
.kpi.hero{grid-column:span 2;background:linear-gradient(135deg,var(--teal),var(--teal-d));color:#fff;border:none}
.kpi.hero .lbl,.kpi.hero .note{color:#e8fbf8}.kpi.hero .val{color:#fff}
.grid2{display:grid;gap:18px;grid-template-columns:1fr 1fr}
@media(max-width:820px){.grid2{grid-template-columns:1fr}.kpi.hero{grid-column:span 1}}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px}
.card img{width:100%;height:auto;display:block;border-radius:8px}
table{width:100%;border-collapse:collapse;font-size:13px;background:var(--card);
border:1px solid var(--line);border-radius:10px;overflow:hidden}
th,td{padding:7px 10px;text-align:right;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}thead th{background:#0f9d8f14;color:var(--muted);
font-size:11px;text-transform:uppercase;letter-spacing:.04em}
tbody tr:last-child td{border-bottom:none}
.dl{display:grid;grid-template-columns:1fr auto;gap:6px 16px;font-size:14px;
background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 18px}
.dl .k{color:var(--muted)}.dl .v{font-weight:600;text-align:right}
.pill{display:inline-block;padding:2px 10px;border-radius:999px;font-size:12px;font-weight:600}
.pill.ok{background:#0f9d8f22;color:var(--teal)}.pill.bad{background:#ef535022;color:var(--red)}
.verdict{background:var(--card);border-left:4px solid var(--teal);border-radius:10px;
padding:14px 18px;font-size:15px;margin:18px 0}
.foot{color:var(--muted);font-size:12px;margin-top:8px}
.recon{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
.recon .kpi .val{font-size:24px}
"""


def _kpi(lbl: str, val: str, note: str = "", cls: str = "") -> str:
    n = f'<div class="note">{note}</div>' if note else ""
    return f'<div class="kpi {cls}"><div class="lbl">{lbl}</div><div class="val">{val}</div>{n}</div>'


def _page(title: str, body: str) -> str:
    return (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{title}</title><style>{_CSS}</style></head>"
            f"<body><div class='wrap'>{body}</div></body></html>")


# ─────────────────────────────────────────────────────────────────────────────
# Executive dashboard
# ─────────────────────────────────────────────────────────────────────────────

def render_executive_dashboard(result: dict, config: FullConfig, data: dict) -> str:
    m = compute_metrics(result, config)
    pay = "Year 1" if m["payback"] == 1 else (f"Year {m['payback']}" if m["payback"] else "—")
    verdict = (f"<b>{m['project_name']}</b> — a {m['S']:.0f} MW solar / {m['W']:.0f} MW wind / "
               f"{m['bess_mwh']:.0f} MWh BESS hybrid delivers <b>₹{m['tod_npv_cr']:,.0f} Cr</b> "
               f"of ToD-aware client savings over 25 years "
               f"(<b>{m['savings_pct_y1']:.0f}%</b> below the DISCOM baseline), "
               f"meeting <b>{m['re_penetration']:.0f}%</b> of load with renewables at "
               f"<b>₹{m['landed_y1']:.2f}/kWh</b> vs the grid's ₹{m['discom_tariff']:.2f}/kWh.")

    kpis = "".join([
        _kpi("Client Savings NPV (25 yr, ToD-aware)", f"₹{m['tod_npv_cr']:,.0f} Cr",
             f"{m['savings_pct_y1']:.0f}% below DISCOM baseline", cls="hero"),
        _kpi("Payback", pay, "client saves from day one"),
        _kpi("Landed vs DISCOM", f"₹{m['landed_y1']:.2f}",
             f"vs grid ₹{m['discom_tariff']:.2f}/kWh (Yr 1)"),
        _kpi("RE Penetration", f"{m['re_penetration']:.0f}%", "of load met by renewables"),
        _kpi("Plant", f"{m['S']:.0f}/{m['W']:.0f} MW",
             f"Solar/Wind · {m['bess_mwh']:.0f} MWh BESS · {m['P']:.0f} MW PPA"),
        _kpi("LCOE", f"₹{m['lcoe']:.2f}", f"CUF {m['plant_cuf']:.0f}% · CAPEX ₹{m['total_capex_cr']:,.0f} Cr"),
    ])
    charts = (f"<div class='grid2'><div class='card'><img src='{_chart_savings(result, m)}'></div>"
              f"<div class='card'><img src='{_chart_tariff(result, m)}'></div></div>")
    body = (f"<h1>{m['project_name']}</h1>"
            f"<div class='sub'>{m['location']} · Executive summary · Pyomo optimal-dispatch model</div>"
            f"<div class='verdict'>{verdict}</div>"
            f"<div class='kpis'>{kpis}</div>{charts}"
            f"<div class='foot'>Headline savings are ToD-aware (optimal dispatch valued at hourly "
            f"Time-of-Day tariffs). Flat-tariff estimate ₹{m['flat_npv_cr']:,.0f} Cr; "
            f"legacy heuristic-controller floor ₹{m['rtc_npv_cr']:,.0f} Cr.</div>")
    return _page(f"{m['project_name']} — Executive Summary", body)


# ─────────────────────────────────────────────────────────────────────────────
# Detailed dashboard
# ─────────────────────────────────────────────────────────────────────────────

def render_detailed_dashboard(result: dict, config: FullConfig, data: dict) -> str:
    m = compute_metrics(result, config)
    fi = result["finance"]
    cap = fi["capex"]
    lcd = fi["lcoe_breakdown"]
    pay = "Year 1" if m["payback"] == 1 else (f"Year {m['payback']}" if m["payback"] else "—")
    vpill = ("<span class='pill ok'>PASS</span>" if m["verify_ok"]
             else "<span class='pill bad'>CHECK</span>")

    band = "".join([
        _kpi("Savings NPV (ToD-aware)", f"₹{m['tod_npv_cr']:,.0f} Cr",
             f"{m['savings_pct_y1']:.1f}% below baseline", cls="hero"),
        _kpi("LCOE", f"₹{m['lcoe']:.3f}", "levelized busbar cost"),
        _kpi("Landed vs DISCOM", f"₹{m['landed_y1']:.2f} / ₹{m['discom_tariff']:.2f}", "Yr-1 / grid"),
        _kpi("Post-solve checks", vpill, "energy-balance invariants"),
    ])

    sizing_dl = "".join(f"<div class='k'>{k}</div><div class='v'>{v}</div>" for k, v in [
        ("Solar (AC MW)", f"{m['S']:.2f}"), ("Wind (MW)", f"{m['W']:.2f}"),
        ("PPA cap (MW)", f"{m['P']:.2f}"), ("BESS containers", f"{m['nb']}"),
        ("BESS energy (MWh)", f"{m['bess_mwh']:.1f}"), ("Charge source", m["charge_source"]),
        ("Plant CUF", f"{m['plant_cuf']:.2f}%"), ("RE penetration", f"{m['re_penetration']:.2f}%"),
    ])
    fin_dl = "".join(f"<div class='k'>{k}</div><div class='v'>{v}</div>" for k, v in [
        ("Total CAPEX", f"₹{m['total_capex_cr']:,.1f} Cr"),
        ("Solar CAPEX", f"₹{_cr(cap['solar_capex']):,.1f} Cr"),
        ("Wind CAPEX", f"₹{_cr(cap['wind_capex']):,.1f} Cr"),
        ("BESS CAPEX", f"₹{_cr(cap['bess_capex']):,.1f} Cr"),
        ("Transmission", f"₹{_cr(cap['transmission_capex']):,.1f} Cr"),
        ("WACC", f"{m['wacc']:.2f}%"),
        ("NPV total cost", f"₹{_cr(lcd['npv_total_cost']):,.1f} Cr"),
        ("Payback", pay),
    ])

    # 25-year table (ToD-aware savings)
    ep = fi["energy_projection"]; lts = fi["landed_tariff_series"]
    opx = fi["opex_projection"]
    tod = np.asarray(result["tod_annual_savings"]); cum = np.cumsum(tod)
    rows = ""
    for y in range(len(tod)):
        rows += (f"<tr><td>{y+1}</td>"
                 f"<td>{ep['delivered_pre_mwh'][y]:,.0f}</td>"
                 f"<td>{ep['delivered_meter_mwh'][y]:,.0f}</td>"
                 f"<td>{opx[y]/CRORE_TO_RS:,.2f}</td>"
                 f"<td>{lts[y]:.3f}</td>"
                 f"<td>{tod[y]/CRORE_TO_RS:,.2f}</td>"
                 f"<td>{cum[y]/CRORE_TO_RS:,.1f}</td></tr>")
    table = (f"<table><thead><tr><th>Yr</th><th>Busbar MWh</th><th>Meter MWh</th>"
             f"<th>OPEX ₹Cr</th><th>Landed ₹/kWh</th><th>Savings ₹Cr</th>"
             f"<th>Cum ₹Cr</th></tr></thead><tbody>{rows}</tbody></table>")

    recon = (f"<div class='recon'>"
             + _kpi("ToD-aware (headline)", f"₹{m['tod_npv_cr']:,.0f} Cr", "optimal dispatch · hourly ToD")
             + _kpi("Flat-tariff estimate", f"₹{m['flat_npv_cr']:,.0f} Cr", "FinanceEngine · ToD-blind")
             + _kpi("RTC heuristic floor", f"₹{m['rtc_npv_cr']:,.0f} Cr", "legacy controller dispatch")
             + "</div>")

    load = data["load_profile"]
    charts1 = (f"<div class='grid2'><div class='card'><img src='{_chart_savings(result, m)}'></div>"
               f"<div class='card'><img src='{_chart_tariff(result, m)}'></div></div>")
    charts2 = (f"<div class='grid2'><div class='card'><img src='{_chart_energy_mix(result)}'></div>"
               f"<div class='card'><img src='{_chart_opex(result)}'></div></div>")
    charts3 = f"<div class='card'><img src='{_chart_dispatch_day(result, load)}'></div>"

    body = (f"<h1>{m['project_name']}</h1>"
            f"<div class='sub'>{m['location']} · Detailed technical dashboard · "
            f"Pyomo optimal-dispatch (single-year MILP)</div>"
            f"<div class='kpis'>{band}</div>"
            f"<h2>Savings & Tariff</h2>{charts1}"
            f"<h2>Economics reconciliation</h2>{recon}"
            f"<div class='foot'>ToD-aware credits hourly Time-of-Day value of delivery; it sits "
            f"below the flat estimate because solar output concentrates in low-ToD hours. The RTC "
            f"floor is what the legacy heuristic controller achieves at the same sizing.</div>"
            f"<h2>Configuration</h2><div class='grid2'><div class='dl'>{sizing_dl}</div>"
            f"<div class='dl'>{fin_dl}</div></div>"
            f"<h2>Energy & OPEX</h2>{charts2}"
            f"<h2>Optimal dispatch (sample day)</h2>{charts3}"
            f"<h2>25-Year projection (ToD-aware savings)</h2>{table}")
    return _page(f"{m['project_name']} — Detailed Dashboard", body)


# ─────────────────────────────────────────────────────────────────────────────
# Writer
# ─────────────────────────────────────────────────────────────────────────────

def write_dashboards(result: dict, config: FullConfig, data: dict,
                     outputs_dir: Path) -> dict[str, Path]:
    """Render both dashboards to outputs_dir; return {name: path}."""
    outputs_dir.mkdir(parents=True, exist_ok=True)
    exec_path = outputs_dir / "dashboard_executive.html"
    det_path  = outputs_dir / "dashboard_detailed.html"
    exec_path.write_text(render_executive_dashboard(result, config, data), encoding="utf-8")
    det_path.write_text(render_detailed_dashboard(result, config, data), encoding="utf-8")
    return {"executive": exec_path, "detailed": det_path}
