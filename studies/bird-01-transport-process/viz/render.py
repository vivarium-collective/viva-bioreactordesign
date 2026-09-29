"""Render the BiRD-01 transport-process split dashboard.

Runs the standalone 0D reactor composite (make_reactor_document) headlessly and
builds a 2x2 interactive Plotly dashboard as a self-contained HTML file.

The study claim: the gas-liquid transport physics is extracted so the standalone
reactor emits only the kLa*(C*-C) transport contribution while biomass is owned
by the cell engine. Baseline composite = make_reactor_document
(viva_bioreactordesign.composites.reactor, standalone 0D reactor).

Plotly's Python package is not installed in this venv (and we must not pip),
so we assemble the Plotly figure spec (traces + layout) as plain dicts and emit
them as JSON into a self-contained HTML that loads plotly.js from the CDN and
calls Plotly.newPlot. All trajectory numbers come from a real composite run.

Run:  <venv>/bin/python render.py   ->  writes bird-01-dashboard.html next to it.
"""

import json
import os

from process_bigraph import Composite
from process_bigraph.emitter import gather_emitter_results
from viva_bioreactordesign.core import build_core
from viva_bioreactordesign.composites import make_reactor_document

# --- design system (shared across the four BiRD study dashboards) -------------
C_O2 = "#2563eb"        # O2 / primary (blue)
C_CO2 = "#f59e0b"       # CO2 / secondary (amber)
C_BIO = "#16a34a"       # biomass (green)
C_KLA = "#7c3aed"       # kLa / transport (violet)
C_UPTAKE = "#dc2626"    # uptake / consumption (red)
C_NEUTRAL = "#94a3b8"   # neutral / reference / grid (slate)
PAPER = "#fbfcfd"
GRID = "#e2e8f0"
FONT = "-apple-system, system-ui, sans-serif"
INK = "#1e293b"

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"

# --- run parameters -----------------------------------------------------------
REACTOR_TYPE = "bubble_column"
INITIAL_BIOMASS_GL = 0.5
INTERVAL_H = 1.0
N_STEPS = 60
COMPOSITE_ID = "make_reactor_document (viva_bioreactordesign.composites.reactor)"


def run_composite():
    """Run the standalone 0D reactor and return per-step emitter rows.

    Adds o2_saturation to the factory's emitter (written to the stores by the
    reactor but not emitted by default) so panel 3 can compute the kLa*(C*-C)
    transport rate. Only the in-memory document is modified; no file outside
    viz/ is touched.
    """
    core = build_core()
    doc = make_reactor_document(
        reactor_type=REACTOR_TYPE,
        initial_biomass_gL=INITIAL_BIOMASS_GL,
        interval=INTERVAL_H,
    )
    doc["emitter"]["config"]["emit"]["o2_saturation"] = "float"
    doc["emitter"]["inputs"]["o2_saturation"] = ["stores", "o2_saturation"]

    sim = Composite({"state": doc}, core=core)
    sim.run(N_STEPS)
    rows = gather_emitter_results(sim)[("emitter",)]

    # Drop the t=0 pre-update row: the emitter fires once before the reactor
    # first writes the stores, so every observable reads 0 there. It is an
    # initialization artifact, not a physical state.
    rows = [r for r in rows if float(r.get("time", 0.0)) > 0.0]
    rows.sort(key=lambda r: float(r["time"]))
    return rows


def col(rows, key):
    return [float(r[key]) for r in rows]


def line(x, y, name, color, axis="y", width=2.5, dash=None, legend="legend"):
    tr = {
        "type": "scatter", "mode": "lines", "name": name,
        "x": x, "y": y, "yaxis": axis,
        "line": {"color": color, "width": width},
        "legend": legend,
        "hovertemplate": f"{name}<br>t=%{{x:.0f}} h<br>%{{y:.3f}}<extra></extra>",
    }
    # map each y axis to its matching x axis (subplot column/row)
    xmap = {"y": "x", "y2": "x2", "y5": "x2", "y3": "x3", "y4": "x4", "y6": "x4"}
    tr["xaxis"] = xmap.get(axis, "x")
    if dash:
        tr["line"]["dash"] = dash
    return tr


def subtitle(text, x, y):
    return {
        "text": f"<b>{text}</b>", "showarrow": False,
        "xref": "paper", "yref": "paper", "x": x, "y": y,
        "xanchor": "center", "yanchor": "bottom",
        "font": {"size": 14, "color": "#0f172a"},
    }


def note(text, xref, yref, x, y, ax=40, ay=-20, arrow=True):
    a = {
        "text": text, "xref": xref, "yref": yref, "x": x, "y": y,
        "showarrow": arrow, "font": {"size": 11, "color": "#475569"},
        "align": "left",
    }
    if arrow:
        a.update({"arrowhead": 2, "arrowcolor": C_NEUTRAL, "arrowwidth": 1,
                  "ax": ax, "ay": ay})
    return a


def build_figure(rows):
    t = col(rows, "time")
    do = col(rows, "dissolved_o2")
    dco2 = col(rows, "dissolved_co2")
    biomass = col(rows, "biomass")
    mu = col(rows, "specific_growth_rate")
    kla = col(rows, "kla_o2")
    holdup = col(rows, "gas_holdup")
    o2_sat = col(rows, "o2_saturation")
    uptake = col(rows, "o2_uptake_rate")
    # transport rate = kLa_O2 * (C* - C)  [mg/(L.h)]
    transport = [k * (s - c) for k, s, c in zip(kla, o2_sat, do)]

    n = len(t)
    mid = n // 2
    do_min = min(do)

    traces = [
        # Panel 1 (x, y)
        line(t, do, "dissolved O₂", C_O2, "y", legend="legend"),
        line(t, dco2, "dissolved CO₂", C_CO2, "y", legend="legend"),
        # Panel 2 (x2, y2 biomass / y5 growth)
        line(t, biomass, "biomass", C_BIO, "y2", legend="legend2"),
        line(t, mu, "growth rate μ", C_NEUTRAL, "y5", width=2,
             dash="dot", legend="legend2"),
        # Panel 3 (x3, y3)
        line(t, transport, "kLa·(C*−C) transport", C_KLA, "y3",
             legend="legend3"),
        line(t, uptake, "O₂ uptake rate", C_UPTAKE, "y3", dash="dash",
             legend="legend3"),
        # Panel 4 (x4, y4 kLa / y6 holdup)
        line(t, kla, "kLaₒ₂", C_KLA, "y4", legend="legend4"),
        line(t, holdup, "gas holdup", C_NEUTRAL, "y6", width=2, dash="dot",
             legend="legend4"),
    ]
    # panel 3 traces belong on x3
    traces[4]["xaxis"] = "x3"
    traces[5]["xaxis"] = "x3"

    # where transport meets/exceeds uptake: at quasi-steady state the supply
    # rate kLa*(C*-C) balances the demand (uptake) to within a fraction of a
    # percent, so detect "meets" with a small relative tolerance.
    meet_idx = next(
        (i for i in range(n)
         if uptake[i] > 0 and transport[i] >= uptake[i] * (1 - 1e-3)),
        None,
    )

    # axis geometry: 2x2 grid, right-side secondary axes on panels 2 & 4
    ax_common = {
        "gridcolor": GRID, "zeroline": False, "linecolor": "#cbd5e1",
        "ticks": "outside", "tickcolor": "#cbd5e1", "showline": True,
    }
    layout = {
        "template": "plotly_white",
        "paper_bgcolor": PAPER, "plot_bgcolor": PAPER,
        "font": {"family": FONT, "size": 12, "color": INK},
        "autosize": True, "height": 720,
        "margin": {"l": 66, "r": 66, "t": 44, "b": 52},
        "hovermode": "x unified",
        # --- panel 1: top-left ---
        "xaxis": {**ax_common, "domain": [0.0, 0.42], "anchor": "y",
                  "title": {"text": "time (h)"}},
        "yaxis": {**ax_common, "domain": [0.58, 1.0], "anchor": "x",
                  "title": {"text": "dissolved gas (mg/L)"}},
        # --- panel 2: top-right (secondary y5) ---
        "xaxis2": {**ax_common, "domain": [0.58, 1.0], "anchor": "y2",
                   "title": {"text": "time (h)"}},
        "yaxis2": {**ax_common, "domain": [0.58, 1.0], "anchor": "x2",
                   "title": {"text": "biomass (g/L)"},
                   "titlefont": {"color": C_BIO}, "tickfont": {"color": C_BIO}},
        "yaxis5": {"domain": [0.58, 1.0], "anchor": "x2", "overlaying": "y2",
                   "side": "right", "zeroline": False, "showgrid": False,
                   "title": {"text": "μ (1/h)"},
                   "titlefont": {"color": "#64748b"},
                   "tickfont": {"color": "#64748b"}},
        # --- panel 3: bottom-left ---
        "xaxis3": {**ax_common, "domain": [0.0, 0.42], "anchor": "y3",
                   "title": {"text": "time (h)"}},
        "yaxis3": {**ax_common, "domain": [0.0, 0.42], "anchor": "x3",
                   "title": {"text": "O₂ rate (mg/(L·h))"}},
        # --- panel 4: bottom-right (secondary y6) ---
        "xaxis4": {**ax_common, "domain": [0.58, 1.0], "anchor": "y4",
                   "title": {"text": "time (h)"}},
        "yaxis4": {**ax_common, "domain": [0.0, 0.42], "anchor": "x4",
                   "title": {"text": "kLaₒ₂ (1/h)"},
                   "titlefont": {"color": C_KLA}, "tickfont": {"color": C_KLA}},
        "yaxis6": {"domain": [0.0, 0.42], "anchor": "x4", "overlaying": "y4",
                   "side": "right", "zeroline": False, "showgrid": False,
                   "title": {"text": "gas holdup (–)"},
                   "titlefont": {"color": "#64748b"},
                   "tickfont": {"color": "#64748b"}},
        # four compact per-panel legends
        "legend": {"x": 0.0, "y": 1.0, "xanchor": "left", "yanchor": "top",
                   "bgcolor": "rgba(251,252,253,0.65)", "font": {"size": 10},
                   "bordercolor": GRID, "borderwidth": 1},
        "legend2": {"x": 0.585, "y": 1.0, "xanchor": "left", "yanchor": "top",
                    "bgcolor": "rgba(251,252,253,0.65)", "font": {"size": 10},
                    "bordercolor": GRID, "borderwidth": 1},
        "legend3": {"x": 0.0, "y": 0.42, "xanchor": "left", "yanchor": "top",
                    "bgcolor": "rgba(251,252,253,0.65)", "font": {"size": 10},
                    "bordercolor": GRID, "borderwidth": 1},
        "legend4": {"x": 0.585, "y": 0.42, "xanchor": "left", "yanchor": "top",
                    "bgcolor": "rgba(251,252,253,0.65)", "font": {"size": 10},
                    "bordercolor": GRID, "borderwidth": 1},
        "annotations": [
            subtitle("Dissolved gases", 0.21, 1.0),
            subtitle("Biomass &amp; growth rate", 0.79, 1.0),
            subtitle("Transport vs consumption (O₂)", 0.21, 0.44),
            subtitle("Transport coefficient &amp; gas holdup", 0.79, 0.44),
            # panel 1
            note(f"O₂ drawn down to ~{do_min:.2f} mg/L<br>as cells "
                 f"consume it; CO₂ accumulates",
                 "x", "y", t[min(4, n - 1)], max(dco2) * 0.5,
                 ax=44, ay=-6),
            # panel 2
            note(f"biomass {biomass[0]:.2f}→{biomass[-1]:.2f} g/L "
                 f"as μ decays<br>(O₂-limited growth)",
                 "x2", "y2", t[mid], biomass[mid], ax=-36, ay=44),
        ],
    }

    if meet_idx is not None:
        layout["annotations"].append(
            note(f"transport ≈ uptake from t≈{t[meet_idx]:.0f} h<br>"
                 f"(supply meets demand: ~{transport[meet_idx]:.0f} "
                 f"mg/(L·h))",
                 "x3", "y3", t[meet_idx], transport[meet_idx], ax=54, ay=42)
        )
    else:
        layout["annotations"].append(
            note("uptake stays above transport<br>(O₂ transfer-limited)",
                 "x3", "y3", t[mid], max(uptake) * 0.6, arrow=False)
        )
    layout["annotations"].append(
        note(f"kLaₒ₂ ≈ {kla[-1]:.1f} 1/h, holdup ≈ "
             f"{holdup[-1]:.3f}<br>(set by geometry &amp; gas flow, steady)",
             "x4", "y4", t[mid], kla[mid] * 0.6, arrow=False)
    )

    stats = dict(
        do_min=do_min, bio0=biomass[0], bio1=biomass[-1],
        kla=kla[-1], holdup=holdup[-1],
        meet_t=(t[meet_idx] if meet_idx is not None else None),
        transport_end=transport[-1], uptake_end=uptake[-1],
    )
    return {"data": traces, "layout": layout}, stats


def build_html(fig, rows):
    fig_json = json.dumps(fig)
    claim = ("Gas–liquid transport physics is extracted: the standalone "
             "reactor emits only the kLa·(C*−C) transport contribution "
             "while biomass is owned by the cell engine.")
    prov = (f"composite <code>{COMPOSITE_ID}</code> &middot; "
            f"reactor_type=<code>{REACTOR_TYPE}</code>, "
            f"initial_biomass_gL=<code>{INITIAL_BIOMASS_GL}</code>, "
            f"interval=<code>{INTERVAL_H} h</code> &middot; "
            f"n_steps=<code>{N_STEPS}</code> &middot; "
            f"emitted rows=<code>{len(rows)}</code>")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BiRD-01 - Transport-process split</title>
<script src="{PLOTLY_CDN}"></script>
<style>
  :root {{ color-scheme: light; }}
  html, body {{ margin: 0; background: {PAPER}; }}
  body {{ font-family: {FONT}; color: {INK}; padding: 16px; }}
  .wrap {{ max-width: 1180px; margin: 0 auto; }}
  header.hd {{ border-left: 4px solid {C_KLA}; padding: 6px 0 10px 14px;
              margin: 4px 0 12px; }}
  header.hd h1 {{ font-size: 20px; margin: 0 0 4px; color: #0f172a;
                 font-weight: 650; }}
  header.hd .claim {{ font-size: 13.5px; margin: 0 0 6px; color: #334155;
                     max-width: 900px; line-height: 1.45; }}
  header.hd .prov {{ font-size: 11.5px; color: #64748b; line-height: 1.5; }}
  header.hd .prov code {{ background: #eef2f7; padding: 1px 5px;
                          border-radius: 4px; color: #334155; }}
  #bird01-dashboard {{ width: 100%; height: 720px; }}
</style>
</head>
<body>
  <div class="wrap">
    <header class="hd">
      <h1>BiRD-01 &middot; Transport-process split</h1>
      <p class="claim">{claim}</p>
      <p class="prov">{prov}</p>
    </header>
    <div id="bird01-dashboard"></div>
  </div>
<script>
  var fig = {fig_json};
  Plotly.newPlot("bird01-dashboard", fig.data, fig.layout,
                 {{responsive: true, displaylogo: false}});
</script>
</body>
</html>
"""


def main():
    rows = run_composite()
    fig, stats = build_figure(rows)
    html = build_html(fig, rows)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "bird-01-dashboard.html")
    with open(out, "w") as f:
        f.write(html)
    size = os.path.getsize(out)
    print(f"wrote {out} ({size/1024:.1f} KB) from {len(rows)} emitted rows")
    print(f"  DO min={stats['do_min']:.3f} mg/L | "
          f"biomass {stats['bio0']:.2f}->{stats['bio1']:.2f} g/L | "
          f"kLa={stats['kla']:.2f} 1/h | holdup={stats['holdup']:.4f}")
    print(f"  transport meets uptake at t={stats['meet_t']} h | "
          f"end transport={stats['transport_end']:.1f}, "
          f"uptake={stats['uptake_end']:.1f} mg/(L.h)")


if __name__ == "__main__":
    main()
