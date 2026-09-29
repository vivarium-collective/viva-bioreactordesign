#!/usr/bin/env python
"""BiRD-03 · Coupling-interval convergence — multi-panel dashboard.

Runs the coupled reactor<->cell composite (make_coupled_document) at several
coupling intervals on a COMMON physical timeline and renders a self-contained
Plotly HTML that demonstrates the study claim:

    the reactor coupling/update interval is a tunable parameter, and the
    coupled dissolved-O2 trajectory CONVERGES as the interval decreases (dt -> 0).

Regime note (important, and why the intervals below are sub-hour):
    The composite integrates the gas->liquid transport term with an explicit
    (forward-Euler) step. Stability requires kLa * dt < ~1. With kLa ~ 11 / h
    the coupled loop is stable only for dt below ~0.09 h; coarse intervals
    (e.g. >=0.25 h) sit in the explicit-Euler OVERSHOOT regime and diverge
    exponentially -- that interval-dependence is exactly what this study
    characterizes, and the reason to keep dt small. So the convergence fan is
    built in the physically meaningful stable regime, mirroring the study's
    validated acceptance test (tests/test_bird03_coupling_interval.py).

Plotly-python is not installed in the workbench venv, so the figure is emitted
as trace/layout JSON and drawn by Plotly.newPlot from the CDN (the spec's
required delivery). Run:  <workbench-venv>/bin/python render.py
Writes bird-03-dashboard.html next to this file.
"""

import os
import json
import numpy as np

from process_bigraph import Composite
from process_bigraph.emitter import gather_emitter_results
from viva_bioreactordesign.core import build_core
from viva_bioreactordesign.composites import make_coupled_document

# --- shared design system (from viz-spec) ---
COL_O2 = "#2563eb"       # blue  (dissolved O2 / primary)
COL_BIO = "#16a34a"      # green (biomass)
COL_UPT = "#dc2626"      # red   (uptake/consumption)
COL_REF = "#94a3b8"      # slate (reference / grid)
PAPER = "#fbfcfd"
FONT = "-apple-system, system-ui, sans-serif"
# blue ramp for the interval fan (coarse -> fine); reference drawn separately
FAN_COLORS = ["#93c5fd", "#60a5fa", "#3b82f6", "#1d4ed8"]

# --- run configuration (common physical timeline) ---
TOTAL_H = 4.0                          # same simulated horizon for every interval
INTERVALS = [0.08, 0.04, 0.02, 0.01]   # coupling dt (h), stable regime kLa*dt<1
REF_INTERVAL = 0.0025                  # fine reference solution (dt -> 0 proxy)
T_EVAL = 2.0                           # fixed time for the scalar convergence metric
KLA_APPROX = 11.0                      # kLa magnitude for the stability note
RUN_KW = dict(initial_biomass_gL=0.1, growth_enabled=True, initial_do_mgL=8.0)


def run_coupled(interval, total_h):
    """Run the coupled composite for `total_h` of global time at `interval`."""
    doc = make_coupled_document(interval=interval, **RUN_KW)
    sim = Composite({"state": doc}, core=build_core())
    sim.run(total_h)                   # sim.run(N) advances N units of global time
    rows = gather_emitter_results(sim)[("emitter",)]
    t = np.array([r["time"] for r in rows], dtype=float)
    do = np.array([r["dissolved_o2"] for r in rows], dtype=float)
    bio = np.array([r["biomass"] for r in rows], dtype=float)
    return t, do, bio


def line(x, y, color, name, axis, width=2.0, dash=None, showlegend=True,
         group=None, hover=""):
    tr = {
        "type": "scatter", "mode": "lines", "x": list(x), "y": list(y),
        "name": name, "line": {"color": color, "width": width},
        "xaxis": axis[0], "yaxis": axis[1], "showlegend": showlegend,
        "hovertemplate": hover,
    }
    if dash:
        tr["line"]["dash"] = dash
    if group:
        tr["legendgroup"] = group
    return tr


def markers(x, y, color, name, axis, symbol="circle", size=9, dash=None,
            width=2.0, showlegend=True, group=None, hover=""):
    tr = {
        "type": "scatter", "mode": "lines+markers", "x": list(x), "y": list(y),
        "name": name, "line": {"color": color, "width": width},
        "marker": {"color": color, "size": size, "symbol": symbol},
        "xaxis": axis[0], "yaxis": axis[1], "showlegend": showlegend,
        "hovertemplate": hover,
    }
    if dash:
        tr["line"]["dash"] = dash
    if group:
        tr["legendgroup"] = group
    return tr


def ann(x, y, text, xref, yref, color=COL_REF, xanchor="left", size=10.5):
    return {
        "x": x, "y": y, "xref": xref, "yref": yref, "text": text,
        "showarrow": False, "align": ("right" if xanchor == "right" else "left"),
        "xanchor": xanchor, "font": {"size": size, "color": color, "family": FONT},
    }


def subtitle(x, text):
    return {
        "x": x, "y": 1.0, "xref": "paper", "yref": "paper", "text": f"<b>{text}</b>",
        "showarrow": False, "xanchor": "center", "yanchor": "bottom",
        "font": {"size": 12.5, "color": "#0f172a", "family": FONT},
    }


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "bird-03-dashboard.html")

    # Reference (finest dt) + the interval sweep.
    t_ref, do_ref, bio_ref = run_coupled(REF_INTERVAL, TOTAL_H)
    runs = {h: run_coupled(h, TOTAL_H) for h in INTERVALS}

    # --- convergence metrics vs the fine reference ---
    do_at_eval_ref = float(np.interp(T_EVAL, t_ref, do_ref))
    scalar_err, rms_err, err_traj = [], [], {}
    for h in INTERVALS:
        t, do, _ = runs[h]
        e = np.abs(do - np.interp(t, t_ref, do_ref))
        err_traj[h] = (t, e)
        rms_err.append(float(np.sqrt(np.mean(e ** 2))))
        scalar_err.append(abs(float(np.interp(T_EVAL, t, do)) - do_at_eval_ref))
    order = float(np.log(scalar_err[-2] / scalar_err[-1]) /
                  np.log(INTERVALS[-2] / INTERVALS[-1]))

    traces = []

    # Panel 1 (x/y): DO convergence fan + reference (dashed)
    for i, h in enumerate(INTERVALS):
        t, do, _ = runs[h]
        traces.append(line(
            t, do, FAN_COLORS[i], f"Δt = {h:g} h", ("x", "y"),
            group=f"g{h}",
            hover="t=%{x:.2f} h<br>DO=%{y:.3f} mg/L<extra>Δt=" + f"{h:g} h</extra>"))
    traces.append(line(
        t_ref, do_ref, "#334155", f"reference Δt = {REF_INTERVAL:g} h",
        ("x", "y"), width=2.4, dash="dash",
        hover="t=%{x:.2f} h<br>DO=%{y:.3f} mg/L<extra>reference</extra>"))

    # Panel 2 (x2/y2): log-log convergence metric + slope-1 guide
    traces.append(markers(
        INTERVALS, scalar_err, COL_O2, f"|DO error| at t={T_EVAL:g} h",
        ("x2", "y2"), group="metric",
        hover="Δt=%{x:g} h<br>err=%{y:.2e} mg/L<extra></extra>"))
    traces.append(markers(
        INTERVALS, rms_err, COL_UPT, "RMS error over trajectory", ("x2", "y2"),
        symbol="diamond", size=7, width=1.8, dash="dot", group="metric",
        hover="Δt=%{x:g} h<br>RMS=%{y:.2e} mg/L<extra></extra>"))
    guide = [scalar_err[-1] * (h / INTERVALS[-1]) for h in INTERVALS]
    traces.append(line(
        INTERVALS, guide, COL_REF, "slope 1 (first order)", ("x2", "y2"),
        width=1.4, dash="dash", hover="slope-1 guide<extra></extra>"))

    # Panel 3 (x3/y3): biomass vs time -- collapses onto one curve
    for i, h in enumerate(INTERVALS):
        t, _, bio = runs[h]
        traces.append(line(
            t, bio, FAN_COLORS[i], f"Δt = {h:g} h", ("x3", "y3"),
            showlegend=False, group=f"g{h}",
            hover="t=%{x:.2f} h<br>biomass=%{y:.4f} g/L<extra>Δt="
                  + f"{h:g} h</extra>"))
    traces.append(line(
        t_ref, bio_ref, COL_BIO, "reference", ("x3", "y3"), width=2.4,
        dash="dash", showlegend=False,
        hover="t=%{x:.2f} h<br>biomass=%{y:.4f} g/L<extra>reference</extra>"))
    bm_end = [float(np.interp(TOTAL_H, tt, bb)) for tt, _, bb in runs.values()]
    spread = max(bm_end) - min(bm_end)

    # Panel 4 (x4/y4): |DO - DO_ref| vs time per interval
    for i, h in enumerate(INTERVALS):
        t, e = err_traj[h]
        traces.append(line(
            t, e, FAN_COLORS[i], f"Δt = {h:g} h", ("x4", "y4"),
            showlegend=False, group=f"g{h}",
            hover="t=%{x:.2f} h<br>|err|=%{y:.3e} mg/L<extra>Δt="
                  + f"{h:g} h</extra>"))

    # --- axis layout: 2x2 grid via domains ---
    ax_common = {"gridcolor": "#e2e8f0", "zeroline": False, "linecolor": "#cbd5e1",
                 "ticks": "outside", "tickcolor": "#cbd5e1"}
    layout = {
        "template": None, "paper_bgcolor": PAPER, "plot_bgcolor": PAPER,
        "font": {"family": FONT, "size": 12, "color": "#1e293b"},
        "autosize": True, "height": 780,
        "margin": {"l": 62, "r": 28, "t": 74, "b": 92},
        "title": {
            "text": "Coupled reactor↔cell · dissolved-O₂ convergence "
                    "under coupling-interval refinement",
            "x": 0.5, "xanchor": "center",
            "font": {"size": 14.5, "color": "#0f172a"}},
        "legend": {"orientation": "h", "yanchor": "bottom", "y": -0.13,
                   "xanchor": "center", "x": 0.5, "font": {"size": 11},
                   "bgcolor": "rgba(255,255,255,0.6)"},
        # top-left
        "xaxis": {**ax_common, "domain": [0.0, 0.44], "anchor": "y",
                  "title": {"text": "time (h)"}},
        "yaxis": {**ax_common, "domain": [0.57, 1.0], "anchor": "x",
                  "title": {"text": "dissolved O₂ (mg/L)"}},
        # top-right (log-log)
        "xaxis2": {**ax_common, "domain": [0.57, 1.0], "anchor": "y2",
                   "type": "log", "title": {"text": "coupling interval Δt (h)"}},
        "yaxis2": {**ax_common, "domain": [0.57, 1.0], "anchor": "x2",
                   "type": "log", "title": {"text": "|DO error| (mg/L)"}},
        # bottom-left
        "xaxis3": {**ax_common, "domain": [0.0, 0.44], "anchor": "y3",
                   "title": {"text": "time (h)"}},
        "yaxis3": {**ax_common, "domain": [0.0, 0.42], "anchor": "x3",
                   "title": {"text": "biomass (g/L)"}},
        # bottom-right
        "xaxis4": {**ax_common, "domain": [0.57, 1.0], "anchor": "y4",
                   "title": {"text": "time (h)"}},
        "yaxis4": {**ax_common, "domain": [0.0, 0.42], "anchor": "x4",
                   "title": {"text": "|DO − DO_ref| (mg/L)"}},
        "annotations": [
            subtitle(0.22, "Convergence fan: dissolved O₂ vs time"),
            subtitle(0.785, "Convergence metric: |DO error| vs coupling Δt"),
            subtitle(0.22, "Biomass vs time (collapses — Δt-independent)"),
            subtitle(0.785, "Where the error lives: |DO − DO_ref| vs time"),
            ann(0.96, 0.92, "coarse Δt overshoots the transient;<br>"
                "the fan tightens onto the reference<br>as Δt→0",
                "x domain", "y domain", xanchor="right"),
            ann(0.05, 0.12, f"error → 0 as Δt → 0<br>"
                f"(empirical order ≈ {order:.1f} between<br>the two finest Δt)",
                "x2 domain", "y2 domain", color="#334155"),
            ann(0.05, 0.9, "curves overlap: growth is Δt-independent<br>"
                f"(spread at t={TOTAL_H:g} h ≈ {spread:.1e} g/L)",
                "x3 domain", "y3 domain", color=COL_BIO),
            ann(0.96, 0.9, "error peaks in the fast transient<br>"
                "and shrinks with each Δt halving",
                "x4 domain", "y4 domain", xanchor="right"),
        ],
    }
    # fix the subplot-title x positions to the four panel centres
    layout["annotations"][0]["x"] = 0.22
    layout["annotations"][1]["x"] = 0.785
    layout["annotations"][2]["x"] = 0.22
    layout["annotations"][2]["y"] = 0.42
    layout["annotations"][3]["x"] = 0.785
    layout["annotations"][3]["y"] = 0.42

    fig_json = json.dumps({"data": traces, "layout": layout})

    steps = {h: int(round(TOTAL_H / h)) for h in INTERVALS}
    prov = ", ".join(f"Δt={h:g}h ({steps[h]} steps)" for h in INTERVALS)
    header = f"""
<div style="max-width:1200px;margin:14px auto 0;padding:0 16px;
            font-family:{FONT};color:#0f172a;">
  <div style="font-size:22px;font-weight:650;letter-spacing:-0.01em;">
    BiRD-03 · Coupling-interval convergence</div>
  <div style="font-size:14px;color:#334155;margin-top:4px;">
    Claim: the reactor coupling/update interval is a tunable parameter, and the
    coupled dissolved-O₂ trajectory <b>converges as the interval decreases
    (Δt → 0)</b>.</div>
  <div style="font-size:12px;color:#64748b;margin-top:6px;line-height:1.5;">
    <b>Provenance</b> · composite
    <code>viva_bioreactordesign.composites.coupled_reactor_cell</code>
    (<code>make_coupled_document</code>, growth on, initial biomass 0.1 g/L,
    initial DO 8.0 mg/L) · common horizon <b>{TOTAL_H:g} h</b> ·
    intervals {prov} · fine reference Δt={REF_INTERVAL:g} h
    ({int(round(TOTAL_H / REF_INTERVAL))} steps).<br>
    <span style="color:#94a3b8;">Sub-hour Δt: the transport term uses an
    explicit step, stable only for kLa·Δt ≲ 1
    (kLa≈{KLA_APPROX:g}/h ⇒ Δt≲ 0.09 h). Coarser intervals
    sit in the explicit-Euler overshoot regime and diverge — the very
    interval-dependence this study characterizes; convergence is shown in the
    stable regime, matching the study's acceptance test.</span></div>
</div>"""

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BiRD-03 Coupling-interval convergence</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>body{{margin:0;background:{PAPER};}}</style>
</head><body>
{header}
<div style="max-width:1200px;margin:0 auto;padding:4px 8px 24px;">
  <div id="bird03-figure"></div>
</div>
<script>
  var fig = {fig_json};
  Plotly.newPlot("bird03-figure", fig.data, fig.layout,
                 {{responsive: true, displayModeBar: true,
                   modeBarButtonsToRemove: ["lasso2d","select2d"]}});
</script>
</body></html>"""

    with open(out, "w") as f:
        f.write(html)

    print(f"wrote {out} ({os.path.getsize(out)} bytes)")
    print(f"reference DO@t={T_EVAL}h = {do_at_eval_ref:.5f} mg/L")
    for h, e, r in zip(INTERVALS, scalar_err, rms_err):
        print(f"  dt={h:g}h steps={steps[h]:4d}  |err|@t={T_EVAL}h={e:.5e}  RMS={r:.5e}")
    print(f"biomass spread at t={TOTAL_H:g}h across intervals = {spread:.3e} g/L")
    print(f"empirical convergence order (two finest) = {order:.2f}")


if __name__ == "__main__":
    main()
