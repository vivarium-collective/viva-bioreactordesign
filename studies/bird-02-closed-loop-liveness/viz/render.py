#!/usr/bin/env python
"""Render the BiRD-02 closed-loop-liveness dashboard.

Runs the coupled reactor<->cell composite (BiRDTransportProcess +
MonodCellProcess) headlessly for two scenarios and builds a 2x2 interactive
Plotly dashboard as a self-contained HTML file. Plotly-python is not installed
in this venv, so the figure is assembled as JSON and drawn client-side via
Plotly.newPlot() using the CDN library (one <script src> in the head).

STUDY CLAIM: When the transport process is coupled to the Monod cell engine at
high biomass, the reactor->cell feedback fires -- dissolved O2 drops below
saturation as consumption outpaces transport -- and recovers when biomass is
removed. Net d[O2]/dt = transport - consumption.

Scenarios (growth held off -> static high-biomass liveness test, per study
design; interval = 0.02 h for numerically stable explicit integration, the
verified setting in tests/test_bird02_liveness.py):
  HIGH    : initial_biomass_gL = 5.0, initial_do = 8.0 mg/L  -> DO pulled below C*
  CONTROL : initial_biomass_gL = 0.1, initial_do = 2.0 mg/L  -> DO recovers upward

Run:  <venv>/bin/python render.py   ->  writes bird-02-dashboard.html next to it.
"""

import json
import os

from process_bigraph import Composite
from process_bigraph.emitter import gather_emitter_results
from viva_bioreactordesign.core import build_core
from viva_bioreactordesign.composites import make_coupled_document

# --- design system (shared across the four BiRD dashboards) ---
COL_O2 = '#2563eb'          # O2 / primary (blue)
COL_CO2 = '#f59e0b'         # CO2 / secondary (amber)
COL_BIOMASS = '#16a34a'     # biomass (green)
COL_TRANSPORT = '#7c3aed'   # kLa / transport (violet)
COL_UPTAKE = '#dc2626'      # uptake / consumption (red)
COL_NEUTRAL = '#94a3b8'     # reference / grid (slate)
BAND_FILL = 'rgba(37,99,235,0.08)'
GRID = 'rgba(148,163,184,0.18)'
AXLINE = '#cbd5e1'
BG = '#fbfcfd'
FONT = '-apple-system, system-ui, sans-serif'

C_STAR = 8.21    # O2 saturation C*_O2 (mg/L), bubble-column @ 298 K
SAT_LOW = 7.0    # lower edge of the air-saturation band
INTERVAL = 0.02  # h per update (stable explicit integration)
DURATION = 1.0   # h simulated -> ~50 emitter rows
COMPOSITE_ID = 'viva_bioreactordesign.composites.coupled_reactor_cell'


def run_scenario(initial_biomass_gL, initial_do_mgL):
    core = build_core()
    doc = make_coupled_document(
        reactor_type='bubble_column',
        initial_biomass_gL=initial_biomass_gL,
        initial_do_mgL=initial_do_mgL,
        growth_enabled=False,   # static biomass: liveness, not growth dynamics
        interval=INTERVAL,
    )
    sim = Composite({'state': doc}, core=core)
    sim.run(DURATION)
    return gather_emitter_results(sim)[('emitter',)]


def col(rows, key):
    return [r[key] for r in rows]


def line_trace(x, y, name, color, axis, dash=None, width=2.5,
               show_legend=True, hover=''):
    ln = {'color': color, 'width': width}
    if dash:
        ln['dash'] = dash
    return {
        'type': 'scatter', 'mode': 'lines', 'name': name,
        'x': x, 'y': y, 'line': ln,
        'xaxis': axis[0], 'yaxis': axis[1],
        'showlegend': show_legend,
        'hovertemplate': hover + '<extra></extra>',
    }


def base_axis(title, domain, anchor, rng=None, color=None):
    ax = {
        'title': {'text': title, 'font': {'size': 12}},
        'domain': domain, 'anchor': anchor,
        'gridcolor': GRID, 'zeroline': False,
        'linecolor': AXLINE, 'showline': True, 'ticks': 'outside',
        'tickcolor': AXLINE, 'tickfont': {'size': 10},
    }
    if rng is not None:
        ax['range'] = rng
    if color is not None:
        ax['title']['font']['color'] = color
        ax['tickfont'] = {'size': 10, 'color': color}
        ax['linecolor'] = color
    return ax


def rect_band(xref, yref, y0, y1):
    return {'type': 'rect', 'xref': xref, 'yref': yref, 'x0': 0, 'x1': 1,
            'y0': y0, 'y1': y1, 'fillcolor': BAND_FILL, 'line': {'width': 0},
            'layer': 'below'}


def hline(xref, yref, y):
    return {'type': 'line', 'xref': xref, 'yref': yref, 'x0': 0, 'x1': 1,
            'y0': y, 'y1': y, 'line': {'color': COL_NEUTRAL, 'width': 1,
                                       'dash': 'dash'}}


def build_fig_json(high, control):
    t_h = col(high, 'time')
    do_h = col(high, 'dissolved_o2')
    t_c = col(control, 'time')
    do_c = col(control, 'dissolved_o2')
    supply = col(high, 'o2_transport_delta')
    demand = [-x for x in col(high, 'o2_exchange_delta')]
    biomass_h = col(high, 'biomass')
    mu_h = col(high, 'specific_growth_rate')
    do_h_ss, do_c_ss = do_h[-1], do_c[-1]

    # 2x2 domains: cols [0,.42]/[.58,1], rows [.57,1]/[0,.43]
    xL, xR = [0.0, 0.42], [0.58, 1.0]
    yT, yB = [0.57, 1.0], [0.0, 0.43]

    data = []
    # P1: DO high
    data.append(line_trace(t_h, do_h, 'dissolved O₂ (high biomass)', COL_O2,
                           ('x', 'y'),
                           hover='t=%{x:.2f} h<br>DO=%{y:.3f} mg/L'))
    # P2: supply vs demand
    data.append(line_trace(t_h, supply, 'O₂ transport (supply)', COL_TRANSPORT,
                           ('x2', 'y2'),
                           hover='t=%{x:.2f} h<br>transport +%{y:.3f} mg/L/step'))
    data.append(line_trace(t_h, demand, 'cell O₂ uptake (demand)', COL_UPTAKE,
                           ('x2', 'y2'), dash='dot',
                           hover='t=%{x:.2f} h<br>uptake %{y:.3f} mg/L/step'))
    # P3: recovery overlay
    data.append(line_trace(t_h, do_h, 'DO — high biomass (5 g/L)', COL_O2,
                           ('x3', 'y3'), show_legend=False,
                           hover='t=%{x:.2f} h<br>DO=%{y:.3f} mg/L'))
    data.append(line_trace(t_c, do_c, 'DO — biomass removed (0.1 g/L)',
                           COL_BIOMASS, ('x3', 'y3'),
                           hover='t=%{x:.2f} h<br>DO=%{y:.3f} mg/L'))
    # P4: biomass + mu (secondary y = y5)
    data.append(line_trace(t_h, biomass_h, 'biomass (g/L)', COL_BIOMASS,
                           ('x4', 'y4'),
                           hover='t=%{x:.2f} h<br>biomass=%{y:.3f} g/L'))
    data.append(line_trace(t_h, mu_h, 'specific growth rate (1/h)', COL_CO2,
                           ('x4', 'y5'), dash='dot',
                           hover='t=%{x:.2f} h<br>μ=%{y:.4f} 1/h'))

    layout = {
        'height': 780, 'autosize': True,
        'paper_bgcolor': BG, 'plot_bgcolor': BG,
        'font': {'family': FONT, 'size': 12, 'color': '#1e293b'},
        'margin': {'l': 70, 'r': 60, 't': 44, 'b': 70},
        'legend': {'orientation': 'h', 'yanchor': 'top', 'y': -0.06,
                   'xanchor': 'center', 'x': 0.5, 'font': {'size': 11}},
        'hovermode': 'x unified',
        'xaxis': base_axis('time (h)', xL, 'y'),
        'yaxis': base_axis('dissolved O₂ (mg/L)', yT, 'x', rng=[0, C_STAR + 0.6]),
        'xaxis2': base_axis('time (h)', xR, 'y2'),
        'yaxis2': base_axis('O₂ flux (mg/L per step)', yT, 'x2'),
        'xaxis3': base_axis('time (h)', xL, 'y3'),
        'yaxis3': base_axis('dissolved O₂ (mg/L)', yB, 'x3', rng=[0, C_STAR + 0.6]),
        'xaxis4': base_axis('time (h)', xR, 'y4'),
        'yaxis4': base_axis('biomass (g/L)', yB, 'x4', color=COL_BIOMASS),
        'yaxis5': {'title': {'text': 'specific growth rate (1/h)',
                             'font': {'size': 12, 'color': COL_CO2}},
                   'domain': yB, 'anchor': 'x4', 'overlaying': 'y4',
                   'side': 'right', 'zeroline': False, 'showgrid': False,
                   'linecolor': COL_CO2, 'showline': True, 'ticks': 'outside',
                   'tickcolor': COL_CO2, 'tickfont': {'size': 10, 'color': COL_CO2}},
        'shapes': [
            rect_band('x domain', 'y', SAT_LOW, C_STAR),
            hline('x domain', 'y', C_STAR),
            rect_band('x3 domain', 'y3', SAT_LOW, C_STAR),
            hline('x3 domain', 'y3', C_STAR),
        ],
    }

    # first time DO drops below the saturation band (high arm)
    below_i = next((i for i, v in enumerate(do_h) if v < SAT_LOW), 1)
    gap0 = max(range(min(6, len(t_h))), key=lambda i: demand[i] - supply[i])

    def subtitle(text, xdom, ytop):
        cx = (xdom[0] + xdom[1]) / 2
        return {'text': text, 'xref': 'paper', 'yref': 'paper', 'x': cx,
                'y': ytop, 'yanchor': 'bottom', 'showarrow': False,
                'font': {'family': FONT, 'size': 13, 'color': '#0f172a'}}

    ann = [
        subtitle('Dissolved O₂ vs time — high biomass (5 g/L)', xL, 1.0),
        subtitle('O₂ supply vs demand — transport vs cell uptake', xR, 1.0),
        subtitle('Recovery — DO with vs without biomass', xL, 0.455),
        subtitle('Biomass & specific growth rate (high-biomass arm)', xR, 0.455),
        # P1 annotations
        {'xref': 'x', 'yref': 'y', 'x': t_h[below_i], 'y': do_h[below_i],
         'text': f'DO drops below saturation<br>(feedback fires) → settles {do_h_ss:.2f} mg/L',
         'showarrow': True, 'arrowhead': 2, 'arrowcolor': COL_UPTAKE,
         'ax': 60, 'ay': -40, 'font': {'size': 11, 'color': COL_UPTAKE}},
        {'xref': 'x domain', 'yref': 'y', 'x': 1, 'y': C_STAR, 'xanchor': 'right',
         'yshift': 9, 'text': 'C*_O₂ = 8.21 mg/L', 'showarrow': False,
         'font': {'size': 10, 'color': COL_NEUTRAL}},
        # P2 annotations
        {'xref': 'x2', 'yref': 'y2', 'x': t_h[gap0], 'y': demand[gap0],
         'text': 'demand ≫ supply early:<br>consumption outpaces transport',
         'showarrow': True, 'arrowhead': 2, 'arrowcolor': COL_UPTAKE,
         'ax': 75, 'ay': -25, 'font': {'size': 11, 'color': COL_UPTAKE}},
        {'xref': 'x2 domain', 'yref': 'y2', 'x': 1, 'y': supply[-1],
         'xanchor': 'right', 'yshift': -26,
         'text': 'balance: transport = uptake<br>(steady state below C*)',
         'showarrow': False, 'font': {'size': 10, 'color': COL_TRANSPORT}},
        # P3 annotations
        {'xref': 'x3', 'yref': 'y3', 'x': t_c[-1], 'y': do_c_ss,
         'text': f'biomass removed → DO recovers<br>toward saturation ({do_c_ss:.2f} mg/L)',
         'showarrow': True, 'arrowhead': 2, 'arrowcolor': COL_BIOMASS,
         'ax': -70, 'ay': -30, 'font': {'size': 11, 'color': COL_BIOMASS}},
        {'xref': 'x3', 'yref': 'y3', 'x': t_h[-1], 'y': do_h_ss,
         'text': f'high biomass → DO held low ({do_h_ss:.2f} mg/L)',
         'showarrow': True, 'arrowhead': 2, 'arrowcolor': COL_O2,
         'ax': -60, 'ay': 30, 'font': {'size': 11, 'color': COL_O2}},
        # P4 annotation
        {'xref': 'x4 domain', 'yref': 'y5', 'x': 1, 'y': mu_h[-1],
         'xanchor': 'right', 'yshift': 20,
         'text': f'μ self-limits as O₂ falls → {mu_h[-1]:.3f} 1/h<br>biomass held constant (growth off)',
         'showarrow': False, 'font': {'size': 10, 'color': COL_CO2}},
    ]
    layout['annotations'] = ann
    return {'data': data, 'layout': layout}


def build_html(fig, high, control):
    do_h_ss = high[-1]['dissolved_o2']
    do_c_ss = control[-1]['dissolved_o2']
    n_steps = len(high) - 1
    prov = (f'{COMPOSITE_ID} · make_coupled_document(bubble_column, '
            f'growth_enabled=False, interval={INTERVAL} h) · '
            f'HIGH biomass=5.0 g/L / CONTROL biomass=0.1 g/L · '
            f'{n_steps} steps ({DURATION} h) · C*_O₂=8.21 mg/L')
    claim = ('Coupling transport to the Monod cell engine at high biomass fires the '
             'reactor→cell feedback: dissolved O₂ drops below saturation as '
             'consumption outpaces transport, and recovers when biomass is removed '
             '(net d[O₂]/dt = transport − consumption).')
    result = (f'High biomass settles DO at {do_h_ss:.2f} mg/L '
              f'({100 * (1 - do_h_ss / C_STAR):.0f}% below C*); '
              f'removing biomass recovers DO to {do_c_ss:.2f} mg/L.')
    fig_json = json.dumps(fig)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BiRD-02 · Closed-loop liveness</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
  html, body {{ margin: 0; padding: 0; background: {BG}; }}
  body {{ font-family: {FONT}; color: #1e293b; padding: 18px 20px 10px; }}
  .wrap {{ max-width: 1200px; margin: 0 auto; }}
  h1 {{ font-size: 20px; margin: 0 0 4px; color: #0f172a; }}
  .claim {{ font-size: 13px; line-height: 1.45; color: #334155; margin: 0 0 6px;
            max-width: 960px; }}
  .result {{ font-size: 12.5px; color: {COL_O2}; font-weight: 600; margin: 0 0 6px; }}
  .prov {{ font-size: 11px; color: #64748b; font-family: ui-monospace,
           SFMono-Regular, Menlo, monospace; word-break: break-word; }}
  #chart {{ width: 100%; height: 780px; margin-top: 6px; }}
</style>
</head>
<body>
  <div class="wrap">
    <h1>BiRD-02 · Closed-loop liveness</h1>
    <p class="claim">{claim}</p>
    <p class="result">{result}</p>
    <p class="prov">{prov}</p>
    <div id="chart"></div>
  </div>
  <script>
    var FIG = {fig_json};
    Plotly.newPlot('chart', FIG.data, FIG.layout,
                   {{responsive: true, displaylogo: false}});
  </script>
</body>
</html>
"""


def main():
    print('Running HIGH-biomass scenario (5.0 g/L)...')
    high = run_scenario(initial_biomass_gL=5.0, initial_do_mgL=8.0)
    print('Running CONTROL scenario (0.1 g/L, depleted start)...')
    control = run_scenario(initial_biomass_gL=0.1, initial_do_mgL=2.0)
    print(f'  high: {len(high)} rows, DO {high[0]["dissolved_o2"]:.2f} '
          f'-> {high[-1]["dissolved_o2"]:.3f} mg/L')
    print(f'  ctrl: {len(control)} rows, DO {control[0]["dissolved_o2"]:.2f} '
          f'-> {control[-1]["dissolved_o2"]:.3f} mg/L')

    fig = build_fig_json(high, control)
    html = build_html(fig, high, control)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'bird-02-dashboard.html')
    with open(out, 'w') as f:
        f.write(html)
    print(f'Wrote {out} ({os.path.getsize(out)} bytes)')


if __name__ == '__main__':
    main()
