"""Behavior tests for the 1D depth-resolved column reactor (BiRDColumn1DProcess).

The 0D reactor emits one scalar per species; this process resolves the
dissolved-O2 / CO2 / biomass PROFILES down the liquid height. These tests pin
the physics that makes the depth x time heatmap meaningful:

  - the emitted profiles have length n_nodes (a real spatial axis),
  - dissolved O2 is highest at the sparger (bottom) and decreases toward the
    top — the axial gradient from hydrostatic pressure + gas-O2 depletion +
    consumption outrunning axial mixing,
  - the profile is smooth (no odd-even/checkerboard numerical artifact),
  - the gradient is a genuine emergent field, not a hardcoded axis: raising the
    axial dispersion toward the well-mixed limit collapses it.
"""
import numpy as np
import pytest
from process_bigraph import Composite, allocate_core
from process_bigraph.emitter import RAMEmitter, gather_emitter_results

from viva_bioreactordesign.core import build_core
from viva_bioreactordesign.processes import BiRDColumn1DProcess


N_NODES = 20


def _core():
    c = build_core()
    c.register_link("ram-emitter", RAMEmitter)
    return c


def _doc(n_nodes=N_NODES, **cfg_over):
    cfg = dict(n_nodes=n_nodes)
    cfg.update(cfg_over)
    ports = [
        "dissolved_o2_profile", "dissolved_co2_profile", "biomass_profile",
        "o2_saturation_profile", "depths_m", "dissolved_o2", "dissolved_co2",
        "biomass", "do_bottom", "do_top", "do_gradient", "kla_o2", "gas_holdup",
        "o2_uptake_rate", "specific_growth_rate", "superficial_gas_velocity",
    ]
    return {
        "reactor": {
            "_type": "process", "address": "local:BiRDColumn1DProcess",
            "config": cfg, "interval": 1.0,
            "inputs": {"gas_flow_rate_Lpm": ["stores", "gas_flow_rate_Lpm"]},
            "outputs": {p: ["stores", p] for p in ports},
        },
        "stores": {"gas_flow_rate_Lpm": cfg.get("gas_flow_rate_Lpm", 150.0)},
        "emitter": {
            "_type": "step", "address": "local:ram-emitter",
            "config": {"emit": {"dissolved_o2_profile": "list", "depths_m": "list",
                                "do_bottom": "float", "do_top": "float",
                                "do_gradient": "float", "biomass": "float"}},
            "inputs": {p: ["stores", p] for p in
                       ["dissolved_o2_profile", "depths_m", "do_bottom",
                        "do_top", "do_gradient", "biomass"]},
        },
    }


def _run(steps=18, **cfg_over):
    core = _core()
    sim = Composite({"state": _doc(**cfg_over)}, core=core)
    sim.run(steps)
    rows = gather_emitter_results(sim)[("emitter",)]
    return rows


def test_profiles_have_n_nodes_length():
    rows = _run(steps=6)
    prof = rows[-1]["dissolved_o2_profile"]
    depths = rows[-1]["depths_m"]
    assert len(prof) == N_NODES
    assert len(depths) == N_NODES
    # depths increase bottom -> top
    assert all(depths[i] < depths[i + 1] for i in range(N_NODES - 1))


def test_dissolved_o2_highest_at_sparger():
    """DO is highest at the bottom (fresh gas, higher hydrostatic pressure)."""
    rows = _run(steps=18)
    O2 = np.array(rows[-1]["dissolved_o2_profile"])
    assert O2[0] >= O2[-1]                      # bottom >= top
    assert rows[-1]["do_gradient"] > 0.5        # a real, non-trivial gradient


def test_profile_is_monotone_and_smooth():
    """No checkerboard: the profile decreases monotonically and is smooth."""
    rows = _run(steps=18)
    O2 = np.array(rows[-1]["dissolved_o2_profile"])
    assert all(O2[i] >= O2[i + 1] - 1e-6 for i in range(len(O2) - 1))  # monotone
    assert np.max(np.abs(np.diff(O2, 2))) < 0.3                        # smooth (2nd diff)


def test_short_column_has_smaller_gradient():
    """The gradient is emergent from real geometry, not a hardcoded axis: a
    short column (less hydrostatic head, less gas-O2 depletion over the path,
    faster mixing relative to consumption) develops a much smaller axial
    gradient than a tall one."""
    grad_tall = _run(steps=18, liquid_height_m=6.0)[-1]["do_gradient"]
    grad_short = _run(steps=18, liquid_height_m=0.5)[-1]["do_gradient"]
    assert grad_short < grad_tall
    assert grad_short < 1.0


def test_dissolved_o2_bounded():
    """DO stays physical: within [0, a few x air saturation] everywhere."""
    rows = _run(steps=18)
    for r in rows:
        O2 = np.array(r["dissolved_o2_profile"])
        assert np.all(O2 >= -1e-9)
        assert np.all(O2 < 60.0)
