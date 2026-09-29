"""BiRD-04 · Reactor geometry & axial structure — dashboard render script.

Runs REAL composite simulations (no fabricated data) and builds a self-contained
2x2 Plotly dashboard at bird-04-dashboard.html next to this file.

The Plotly figure spec (traces + layout) is assembled as plain Python dicts and
serialized to JSON, then rendered client-side by plotly.js loaded from the CDN
(one <script src> in the HTML head) via Plotly.newPlot. No plotly Python package
is required.

Two data sources, both real:
  1) 0D geometry comparison via make_reactor_document(reactor_type=...) for
     bubble_column / stirred_tank / airlift — emits geometry-dependent kla_o2.
  2) The 1D depth-resolved column via make_column1d_document(...), whose emitter
     accumulates per-tick depth PROFILES stacked into (time, depth) arrays for
     the headline depth x time heatmaps.

Run:
    /Users/eranagmon/code/viva-bioreactordesign--workbench/.venv/bin/python render.py
"""
from __future__ import annotations

import json
import os

import numpy as np

from process_bigraph import Composite
from process_bigraph.emitter import gather_emitter_results
from viva_bioreactordesign.core import build_core
from viva_bioreactordesign.composites import (
    make_reactor_document,
    make_column1d_document,
)

# ---------------------------------------------------------------------------
# Design system (shared across the four BiRD dashboards)
# ---------------------------------------------------------------------------
PALETTE = {
    'o2': '#2563eb',      # O2 / primary (blue)
    'co2': '#f59e0b',     # CO2 / secondary (amber)
    'biomass': '#16a34a',  # biomass (green)
    'kla': '#7c3aed',     # kLa / transport (violet)
    'uptake': '#dc2626',  # uptake / consumption (red)
    'neutral': '#94a3b8',  # neutral / reference / grid (slate)
}
GEOM_COLORS = {
    'bubble_column': PALETTE['o2'],
    'stirred_tank': PALETTE['kla'],
    'airlift': PALETTE['biomass'],
}
GEOM_LABELS = {
    'bubble_column': 'Bubble column',
    'stirred_tank': 'Stirred tank (Van’t Riet)',
    'airlift': 'Airlift',
}
PAPER_BG = '#fbfcfd'
FONT_FAMILY = '-apple-system, system-ui, sans-serif'

# Run parameters (provenance).
GEOM_STEPS = 60
COLUMN_N_NODES = 20
COLUMN_HEIGHT_M = 6.0
COLUMN_STEPS = 30
IMPELLER_POWER_W = 500.0

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_HTML = os.path.join(HERE, 'bird-04-dashboard.html')


def _f(arr):
    """numpy -> plain python list of floats (JSON-serializable)."""
    return [float(x) for x in np.asarray(arr).ravel()]


def _f2(mat):
    """2D numpy -> list of lists of floats."""
    return [[float(x) for x in row] for row in np.asarray(mat)]


# ---------------------------------------------------------------------------
# Data source 1 — 0D geometry comparison (geometry-selectable kLa claim)
# ---------------------------------------------------------------------------
def run_geometry_comparison(core):
    """Run the standalone reactor for each geometry; return {geom: (t, kla)}."""
    out = {}
    for geom in ('bubble_column', 'stirred_tank', 'airlift'):
        extra = {'impeller_power_W': IMPELLER_POWER_W} if geom == 'stirred_tank' else {}
        doc = make_reactor_document(
            reactor_type=geom, interval=1.0, initial_biomass_gL=0.5, **extra,
        )
        sim = Composite({'state': doc}, core=core)
        sim.run(GEOM_STEPS)
        rows = gather_emitter_results(sim)[('emitter',)]
        # Skip the t=0 pre-fire emit (kla not yet computed).
        rows = [r for r in rows if r.get('time', 0.0) > 0.0]
        t = np.array([r['time'] for r in rows])
        kla = np.array([r['kla_o2'] for r in rows])
        out[geom] = (t, kla)
    return out


# ---------------------------------------------------------------------------
# Data source 2 — 1D depth-resolved column (the heatmaps)
# ---------------------------------------------------------------------------
def run_column(core):
    """Run the axial column; stack per-tick profiles into (time, depth) arrays."""
    doc = make_column1d_document(
        n_nodes=COLUMN_N_NODES, liquid_height_m=COLUMN_HEIGHT_M,
    )
    sim = Composite({'state': doc}, core=core)
    sim.run(COLUMN_STEPS)
    rows = gather_emitter_results(sim)[('emitter',)]
    # Keep only rows whose depth profiles are fully populated (drops the t=0
    # pre-fire emit, whose profile lists are still empty).
    rows = [r for r in rows if len(r.get('dissolved_o2_profile', [])) == COLUMN_N_NODES]

    times = np.array([r['time'] for r in rows])                     # (T,)
    do = np.array([r['dissolved_o2_profile'] for r in rows])        # (T, depth)
    biomass = np.array([r['biomass_profile'] for r in rows])        # (T, depth)
    depths = np.array(rows[-1]['depths_m'])                          # (depth,), bottom->top
    do_gradient = np.array([r['do_gradient'] for r in rows])        # (T,)
    return {
        'times': times, 'do': do, 'biomass': biomass, 'depths': depths,
        'do_gradient': do_gradient,
    }


# ---------------------------------------------------------------------------
# Figure assembly (Plotly figure spec as plain dicts)
# ---------------------------------------------------------------------------
def build_figure(geom, col):
    traces = []

    # -- Panel 1 (x/y): kLa vs time for the three geometries -----------------
    for geom_name in ('bubble_column', 'stirred_tank', 'airlift'):
        t, kla = geom[geom_name]
        traces.append({
            'type': 'scatter', 'mode': 'lines', 'xaxis': 'x', 'yaxis': 'y',
            'x': _f(t), 'y': _f(kla), 'name': GEOM_LABELS[geom_name],
            'legendgroup': 'geom',
            'line': {'color': GEOM_COLORS[geom_name], 'width': 2.5},
            'hovertemplate': f'{GEOM_LABELS[geom_name]}<br>t=%{{x}} h<br>'
                             'kLa=%{y:.1f} 1/h<extra></extra>',
        })

    # -- Panel 2 (x2/y2): DO depth x time heatmap (headline) -----------------
    traces.append({
        'type': 'heatmap', 'xaxis': 'x2', 'yaxis': 'y2',
        'x': _f(col['times']), 'y': _f(col['depths']), 'z': _f2(col['do'].T),
        'colorscale': 'Viridis',
        'zmin': float(col['do'].min()), 'zmax': float(col['do'].max()),
        'colorbar': {'title': {'text': 'DO (mg/L)', 'side': 'right'},
                     'len': 0.42, 'y': 0.79, 'x': 1.005, 'thickness': 14},
        'hovertemplate': 't=%{x} h<br>height=%{y:.2f} m<br>DO=%{z:.2f} mg/L<extra></extra>',
    })

    # -- Panel 3 (x3/y3): biomass depth x time heatmap -----------------------
    traces.append({
        'type': 'heatmap', 'xaxis': 'x3', 'yaxis': 'y3',
        'x': _f(col['times']), 'y': _f(col['depths']), 'z': _f2(col['biomass'].T),
        'colorscale': 'YlGn',
        'zmin': float(col['biomass'].min()), 'zmax': float(col['biomass'].max()),
        'colorbar': {'title': {'text': 'Biomass (g/L)', 'side': 'right'},
                     'len': 0.42, 'y': 0.21, 'x': 1.005, 'thickness': 14},
        'hovertemplate': 't=%{x} h<br>height=%{y:.2f} m<br>biomass=%{z:.3f} g/L<extra></extra>',
    })

    # -- Panel 4 (x4/y4): axial O2 gradient (bottom - top) vs time -----------
    traces.append({
        'type': 'scatter', 'mode': 'lines', 'xaxis': 'x4', 'yaxis': 'y4',
        'x': _f(col['times']), 'y': _f(col['do_gradient']),
        'name': 'DO gradient (bottom − top)', 'legendgroup': 'grad',
        'line': {'color': PALETTE['o2'], 'width': 2.5},
        'fill': 'tozeroy', 'fillcolor': 'rgba(37,99,235,0.10)',
        'hovertemplate': 't=%{x} h<br>ΔDO=%{y:.3f} mg/L<extra></extra>',
    })

    # -- Annotations ---------------------------------------------------------
    st_kla = float(geom['stirred_tank'][1][-1])
    bc_kla = float(geom['bubble_column'][1][-1])
    grad = col['do_gradient']
    grad_max = float(np.max(grad))
    t_at_max = float(col['times'][int(np.argmax(grad))])
    mid_t = float(col['times'][int(len(col['times']) * 0.5)])
    bottom_h = float(col['depths'][0])

    subplot_titles = [
        ('kLa_O₂ by reactor geometry (0D, config-selectable)', 0.0, 1.0, 'left'),
        ('Dissolved O₂ — depth × time (headline)', 0.56, 1.0, 'left'),
        ('Biomass — depth × time', 0.0, 0.46, 'left'),
        ('Axial O₂ gradient (bottom − top)', 0.56, 0.46, 'left'),
    ]
    annotations = []
    for text, x, y, anchor in subplot_titles:
        annotations.append({
            'text': f'<b>{text}</b>', 'xref': 'paper', 'yref': 'paper',
            'x': x, 'y': y, 'xanchor': anchor, 'yanchor': 'bottom',
            'showarrow': False, 'font': {'size': 13, 'color': '#1e293b'},
        })
    # Panel 1 callout.
    annotations.append({
        'xref': 'x', 'yref': 'y',
        'x': float(geom['stirred_tank'][0][-1]), 'y': st_kla,
        'text': f'Stirred tank ≈ {st_kla:.0f} 1/h<br>({st_kla / bc_kla:.1f}× bubble column)',
        'showarrow': True, 'arrowhead': 2, 'arrowcolor': PALETTE['kla'],
        'ax': -70, 'ay': -20, 'font': {'size': 10, 'color': PALETTE['kla']},
        'align': 'left',
    })
    # Panel 2 sparger callout.
    annotations.append({
        'xref': 'x2', 'yref': 'y2', 'x': mid_t, 'y': bottom_h,
        'text': 'sparger-fed high-DO zone', 'showarrow': True, 'arrowhead': 2,
        'arrowcolor': '#ffffff', 'ax': 0, 'ay': -26,
        'font': {'size': 10, 'color': '#ffffff'},
        'bgcolor': 'rgba(37,99,235,0.78)', 'bordercolor': '#ffffff', 'borderwidth': 1,
    })
    # Panel 4 peak callout.
    annotations.append({
        'xref': 'x4', 'yref': 'y4', 'x': t_at_max, 'y': grad_max,
        'text': f'peak ΔDO ≈ {grad_max:.2f} mg/L<br>(bottom richer than top)',
        'showarrow': True, 'arrowhead': 2, 'arrowcolor': PALETTE['o2'],
        'ax': 40, 'ay': -24, 'font': {'size': 10, 'color': PALETTE['o2']},
        'align': 'left',
    })

    # -- Layout: 2x2 grid via axis domains -----------------------------------
    axis_common = {'gridcolor': '#e2e8f0', 'zerolinecolor': '#e2e8f0',
                   'linecolor': '#cbd5e1', 'title_font': {'size': 11}}
    layout = {
        'paper_bgcolor': PAPER_BG, 'plot_bgcolor': PAPER_BG,
        'font': {'family': FONT_FAMILY, 'size': 12, 'color': '#1e293b'},
        'height': 760, 'autosize': True,
        'margin': {'l': 70, 'r': 96, 't': 70, 'b': 56},
        'legend': {'orientation': 'h', 'yanchor': 'bottom', 'y': 1.055,
                   'xanchor': 'left', 'x': 0.0, 'font': {'size': 10}},
        'showlegend': True,
        'annotations': annotations,
        # Panel 1
        'xaxis': {'domain': [0.0, 0.44], 'anchor': 'y', 'title': {'text': 'time (h)'}, **axis_common},
        'yaxis': {'domain': [0.58, 1.0], 'anchor': 'x', 'title': {'text': 'kLa_O₂ (1/h)'}, **axis_common},
        # Panel 2 (heatmap: sparger at bottom of axis, surface at top -> ascending height)
        'xaxis2': {'domain': [0.56, 1.0], 'anchor': 'y2', 'title': {'text': 'time (h)'}, **axis_common},
        'yaxis2': {'domain': [0.58, 1.0], 'anchor': 'x2', 'title': {'text': 'height above sparger (m)'},
                   'autorange': True, **axis_common},
        # Panel 3
        'xaxis3': {'domain': [0.0, 0.44], 'anchor': 'y3', 'title': {'text': 'time (h)'}, **axis_common},
        'yaxis3': {'domain': [0.0, 0.42], 'anchor': 'x3', 'title': {'text': 'height above sparger (m)'},
                   'autorange': True, **axis_common},
        # Panel 4
        'xaxis4': {'domain': [0.56, 1.0], 'anchor': 'y4', 'title': {'text': 'time (h)'}, **axis_common},
        'yaxis4': {'domain': [0.0, 0.42], 'anchor': 'x4', 'title': {'text': 'Δ dissolved O₂ (mg/L)'}, **axis_common},
    }
    return {'data': traces, 'layout': layout}


# ---------------------------------------------------------------------------
# HTML shell
# ---------------------------------------------------------------------------
def write_html(fig, geom, col):
    n_frames = len(col['times'])
    fig_json = json.dumps(fig)
    header = f"""
    <div style="max-width:1180px;margin:0 auto 6px;padding:14px 20px 8px;
                font-family:{FONT_FAMILY};color:#1e293b;">
      <div style="font-size:20px;font-weight:700;letter-spacing:-0.01em;">
        BiRD-04 &middot; Reactor geometry &amp; axial structure</div>
      <div style="font-size:13px;color:#475569;margin-top:3px;line-height:1.45;">
        The reactor supports a stirred-tank kLa correlation alongside the
        bubble-column form, <b>selectable by configuration</b> &mdash; and a
        depth-resolved column shows how dissolved O&#8322; and biomass evolve in
        <b>time and space</b>.</div>
      <div style="font-size:11.5px;color:#64748b;margin-top:6px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;">
        0D geometries: make_reactor_document(bubble_column | stirred_tank
        [P={IMPELLER_POWER_W:.0f} W] | airlift), {GEOM_STEPS} steps &nbsp;&bull;&nbsp;
        column: make_column1d_document(n_nodes={COLUMN_N_NODES},
        liquid_height_m={COLUMN_HEIGHT_M:g}), {COLUMN_STEPS} steps &rarr;
        {n_frames}&times;{COLUMN_N_NODES} depth&times;time profiles</div>
    </div>
    """
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BiRD-04 Reactor geometry &amp; axial structure</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
  html,body{{margin:0;background:{PAPER_BG};}}
  #bird04-dashboard{{max-width:1180px;margin:0 auto;padding:0 12px 20px;}}
</style></head>
<body>
{header}
<div id="bird04-dashboard"></div>
<script>
  var fig = {fig_json};
  Plotly.newPlot('bird04-dashboard', fig.data, fig.layout,
                 {{responsive: true, displaylogo: false}});
</script>
</body></html>"""
    with open(OUT_HTML, 'w') as f:
        f.write(doc)


def main():
    core = build_core()
    geom = run_geometry_comparison(core)
    col = run_column(core)
    fig = build_figure(geom, col)
    write_html(fig, geom, col)

    size = os.path.getsize(OUT_HTML)
    print(f'wrote {OUT_HTML} ({size / 1024:.1f} KB)')
    print(f"  DO heatmap  z shape (depth,time): {col['do'].T.shape}")
    print(f"  biomass heatmap z shape (depth,time): {col['biomass'].T.shape}")
    print(f"  depths (m) bottom->top: {np.round(col['depths'], 3).tolist()}")
    for g in ('bubble_column', 'stirred_tank', 'airlift'):
        t, kla = geom[g]
        print(f"  {g}: final kLa_O2 = {kla[-1]:.2f} 1/h")


if __name__ == '__main__':
    main()
