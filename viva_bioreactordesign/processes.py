"""BioReactorDesign (BiRD) process-bigraph wrappers.

Three processes share one transport module (`transport.py`):

  - BiRDReactorProcess   — gas-liquid transport + internal Monod biomass ODE.
                           Standalone, single-process reactor. Owns biomass.
  - MonodCellProcess     — the biomass/consumption term factored out as a
                           cell-side engine conforming to the cell-side
                           interface contract (the trivial fixture for the
                           coupled studies; substitutable for dFBA / WCM).
  - BiRDTransportProcess — transport ONLY. Takes biomass / substrate /
                           dissolved-gas concentrations as read-only inputs
                           and emits only the gas-liquid transport
                           contribution as additive deltas. Owns no biomass.

The split lets a cell engine and the reactor compose at shared dissolved-
species stores: under process-bigraph update aggregation, the cell's exchange
deltas and the reactor's transport deltas merge additively (bare `float`
ports, never `overwrite[float]`) without either process owning the other's
state.

Time units: hours. Concentrations: mg/L (gas species) or g/L (biomass).
"""

import math
import numpy as np
from scipy.integrate import solve_ivp
from process_bigraph import Process

# Re-export the transport physics from the shared module so existing imports
# (`from viva_bioreactordesign.processes import henry_constant`, etc.) keep working.
from viva_bioreactordesign.transport import (  # noqa: F401
    SPECIES_DATA,
    WC_PSI,
    WC_M,
    RHO_LIQ,
    G,
    water_viscosity,
    henry_constant,
    wilke_chang_diffusivity,
    saturation_concentration,
    higbie_kla,
    bubble_rise_velocity,
    gas_holdup,
    slip_velocity,
    superficial_gas_velocity,
    compute_transport_state,
    o2_transport_rate,
    co2_transport_rate,
)


def monod_kinetics(C_O2, X, cfg):
    """Monod biomass/consumption term (the cell side of the reactor ODE).

    Shared by BiRDReactorProcess (internal biomass) and MonodCellProcess
    (factored-out cell engine) so both compute identical kinetics.

    `cfg` is any mapping with: max_growth_rate_per_h, ks_oxygen_mgL,
    yield_biomass_o2, maintenance_coeff_per_h, respiratory_quotient.

    Returns dict: mu (1/h), q_o2 (mg_O2/(g_X·h)), OUR (mg/(L·h)), CER (mg/(L·h)).
    """
    C_O2_pos = max(C_O2, 0.0)
    mu = cfg['max_growth_rate_per_h'] * C_O2_pos / (
        cfg['ks_oxygen_mgL'] + C_O2_pos)

    # Specific O2 uptake rate (mg_O2 / (g_X · h))
    q_o2 = mu / max(cfg['yield_biomass_o2'], 1e-12) + cfg['maintenance_coeff_per_h']

    OUR = q_o2 * max(X, 0.0) * 1000.0  # mg/(L·h)
    RQ = cfg['respiratory_quotient']
    CER = RQ * OUR * (44.01 / 32.0)    # mg_CO2/(L·h)

    return {'mu': mu, 'q_o2': q_o2, 'OUR': OUR, 'CER': CER}


class BiRDReactorProcess(Process):
    """0D bioreactor: gas-liquid transport + internal Monod biomass ODE.

    Standalone, single-process reactor. Behavior is preserved exactly across
    the transport-module refactor: the transport coefficients come from
    `transport.compute_transport_state` and the kinetics from `monod_kinetics`,
    which are verbatim extractions of the pre-refactor inline computation.

    Time units: hours. All rates in 1/h, concentrations in mg/L or g/L.
    """

    config_schema = {
        # Reactor geometry
        'reactor_type': {'_type': 'string', '_default': 'bubble_column'},
        'volume_L': {'_type': 'float', '_default': 20.0},
        'diameter_m': {'_type': 'float', '_default': 0.2},
        'liquid_height_m': {'_type': 'float', '_default': 0.5},
        # Operating conditions
        'gas_flow_rate_Lpm': {'_type': 'float', '_default': 1.0},
        'temperature_K': {'_type': 'float', '_default': 298.15},
        'pressure_atm': {'_type': 'float', '_default': 1.0},
        'o2_fraction_inlet': {'_type': 'float', '_default': 0.21},
        'co2_fraction_inlet': {'_type': 'float', '_default': 0.0004},
        # Bubble properties
        'mean_bubble_diameter_mm': {'_type': 'float', '_default': 3.0},
        # Microbial kinetics
        'initial_biomass_gL': {'_type': 'float', '_default': 0.5},
        'initial_do_mgL': {'_type': 'float', '_default': 8.0},
        'initial_dco2_mgL': {'_type': 'float', '_default': 0.5},
        'max_growth_rate_per_h': {'_type': 'float', '_default': 0.4},
        'ks_oxygen_mgL': {'_type': 'float', '_default': 0.2},
        'yield_biomass_o2': {'_type': 'float', '_default': 1.2},
        'maintenance_coeff_per_h': {'_type': 'float', '_default': 0.01},
        'respiratory_quotient': {'_type': 'float', '_default': 1.0},
        # Stirred tank specific
        'impeller_power_W': {'_type': 'float', '_default': 0.0},
        # kLa correlation: 'auto' resolves by geometry (vant_riet for stirred_tank,
        # higbie otherwise); 'higbie' (penetration theory) | 'vant_riet' (stirred-tank)
        # force a specific correlation regardless of geometry.
        'kla_correlation': {'_type': 'string', '_default': 'auto'},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self._state = None  # internal ODE state [C_O2, C_CO2, X]
        self._derived = {}  # derived quantities from last update

    def inputs(self):
        return {
            'gas_flow_rate_Lpm': 'float',
        }

    def outputs(self):
        return {
            'dissolved_o2': 'overwrite[float]',
            'dissolved_co2': 'overwrite[float]',
            'biomass': 'overwrite[float]',
            'gas_holdup': 'overwrite[float]',
            'kla_o2': 'overwrite[float]',
            'kla_co2': 'overwrite[float]',
            'bubble_diameter_mm': 'overwrite[float]',
            'o2_saturation': 'overwrite[float]',
            'co2_saturation': 'overwrite[float]',
            'specific_growth_rate': 'overwrite[float]',
            'o2_uptake_rate': 'overwrite[float]',
            'co2_evolution_rate': 'overwrite[float]',
            'superficial_gas_velocity': 'overwrite[float]',
        }

    def _compute_derived(self, C_O2, C_CO2, X, gas_flow_Lpm):
        """Combine shared transport coefficients + shared Monod kinetics.

        Identical in value to the pre-refactor inline computation: the
        transport half is `compute_transport_state`, the cell half is
        `monod_kinetics`.
        """
        t = compute_transport_state(self.config, gas_flow_Lpm)
        k = monod_kinetics(C_O2, X, self.config)
        return {**t, **k}

    def _ode_rhs(self, t, y, gas_flow_Lpm):
        """RHS of the ODE: y = [C_O2 (mg/L), C_CO2 (mg/L), X (g/L)], dy/dt per hour."""
        C_O2, C_CO2, X = y
        C_O2 = max(C_O2, 0.0)
        C_CO2 = max(C_CO2, 0.0)
        X = max(X, 0.0)

        d = self._compute_derived(C_O2, C_CO2, X, gas_flow_Lpm)

        # dC_O2/dt = kLa·(C* − C_O2) − OUR        (transport − consumption)
        dC_O2 = o2_transport_rate(d['kla_o2'], d['cstar_o2'], C_O2) - d['OUR']
        # dC_CO2/dt = CER + transport stripping
        dC_CO2 = d['CER'] + co2_transport_rate(d['kla_co2'], d['cstar_co2'], C_CO2)
        # dX/dt = mu·X
        dX = d['mu'] * X

        return [dC_O2, dC_CO2, dX]

    def initial_state(self):
        cfg = self.config
        C_O2 = cfg['initial_do_mgL']
        C_CO2 = cfg['initial_dco2_mgL']
        X = cfg['initial_biomass_gL']
        self._state = np.array([C_O2, C_CO2, X])

        d = self._compute_derived(C_O2, C_CO2, X, cfg['gas_flow_rate_Lpm'])
        self._derived = d

        return {
            'dissolved_o2': C_O2,
            'dissolved_co2': C_CO2,
            'biomass': X,
            'gas_holdup': d['alpha_gas'],
            'kla_o2': d['kla_o2'],
            'kla_co2': d['kla_co2'],
            'bubble_diameter_mm': cfg['mean_bubble_diameter_mm'],
            'o2_saturation': d['cstar_o2'],
            'co2_saturation': d['cstar_co2'],
            'specific_growth_rate': d['mu'],
            'o2_uptake_rate': d['OUR'],
            'co2_evolution_rate': d['CER'],
            'superficial_gas_velocity': d['Ug'],
        }

    def update(self, state, interval):
        if self._state is None:
            self.initial_state()

        cfg = self.config
        gas_flow = state.get('gas_flow_rate_Lpm', cfg['gas_flow_rate_Lpm'])

        sol = solve_ivp(
            self._ode_rhs,
            [0.0, interval],
            self._state.tolist(),
            method='RK45',
            args=(gas_flow,),
            rtol=1e-8,
            atol=1e-10,
            max_step=interval / 10.0,
        )

        if sol.success:
            self._state = np.array([
                max(sol.y[0, -1], 0.0),
                max(sol.y[1, -1], 0.0),
                max(sol.y[2, -1], 0.0),
            ])
        else:
            dy = self._ode_rhs(0, self._state.tolist(), gas_flow)
            self._state = np.maximum(self._state + np.array(dy) * interval, 0.0)

        C_O2, C_CO2, X = self._state
        d = self._compute_derived(C_O2, C_CO2, X, gas_flow)
        self._derived = d

        return {
            'dissolved_o2': float(C_O2),
            'dissolved_co2': float(C_CO2),
            'biomass': float(X),
            'gas_holdup': d['alpha_gas'],
            'kla_o2': d['kla_o2'],
            'kla_co2': d['kla_co2'],
            'bubble_diameter_mm': cfg['mean_bubble_diameter_mm'],
            'o2_saturation': d['cstar_o2'],
            'co2_saturation': d['cstar_co2'],
            'specific_growth_rate': d['mu'],
            'o2_uptake_rate': d['OUR'],
            'co2_evolution_rate': d['CER'],
            'superficial_gas_velocity': d['Ug'],
        }


class MonodCellProcess(Process):
    """Monod-growth cell engine — the biomass/consumption term factored out.

    Conforms (in simplified flat form) to the cell-side interface contract:
    it OWNS biomass and exposes the three contract outputs —
    ``cell_mass`` (g/L), ``external_exchange_fluxes`` (per-species specific
    rates in **mmol/(gDW·h)**, the contract's standard unit shared with dFBA /
    v2ecoli / OxidizeME), and ``growth_rate`` (1/h). It reads dissolved O2 from
    the reactor and emits its O2/CO2 exchange and its biomass growth as additive
    deltas so it composes with BiRDTransportProcess at the shared stores.

    Biomass ``X`` (g/L) is interpreted as **gDW/L**, so the published molar
    fluxes are apples-to-apples with a gDW-based engine substituted under the
    contract. The Monod ``q_o2`` is mass-specific (g_O2/(gDW·h)) internally; the
    contract field converts it to molar via the species molecular weight.

    Trivial by design: it is the conforming fixture that makes the contract
    concrete and the substitution target for dFBA / whole-cell engines.

    NOTE: in this composite the molar ``external_exchange_fluxes`` is the contract
    SURFACE (what a conforming engine publishes); the actual O2/CO2 coupling here
    is the simplified mg/L delta path written directly to the shared dissolved
    stores. The molar-flux → mg/L dC/dt coupler is the reactor-side coupler that
    lives with the consuming engine (e.g. v2ecoli mbp-03), not in this fixture.

    The full contract publishes fluxes under ``agents.*.metabolism...``; this
    flat fixture exposes them directly and would adapt at a coupler boundary
    for an agent-structured engine (see cell_side_interface_contract.md).
    """

    config_schema = {
        'initial_biomass_gL': {'_type': 'float', '_default': 0.5},
        'max_growth_rate_per_h': {'_type': 'float', '_default': 0.4},
        'ks_oxygen_mgL': {'_type': 'float', '_default': 0.2},
        'yield_biomass_o2': {'_type': 'float', '_default': 1.2},
        'maintenance_coeff_per_h': {'_type': 'float', '_default': 0.01},
        'respiratory_quotient': {'_type': 'float', '_default': 1.0},
        # When False, biomass is held at initial_biomass_gL (static high-density
        # fixture for the closed-loop liveness study; no growth dynamics needed).
        'growth_enabled': {'_type': 'boolean', '_default': True},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self._X = None  # internal biomass state (g/L) — the cell owns biomass

    def inputs(self):
        # Reactor → engine: dissolved O2 drives growth + uptake.
        return {
            'dissolved_o2': 'float',
        }

    def outputs(self):
        return {
            # Additive deltas to shared stores (bare float → compose with transport).
            'biomass': 'float',
            'dissolved_o2': 'float',
            'dissolved_co2': 'float',
            # Cell-side contract readouts (diagnostics; overwrite each step).
            'cell_mass': 'overwrite[float]',
            'growth_rate': 'overwrite[float]',
            'external_exchange_fluxes': 'overwrite[map[float]]',
            # Applied exchange deltas this step (the actual values written to the
            # shared stores, post-cap) — let a coupler / test close the O2 balance.
            'o2_exchange_delta': 'overwrite[float]',
            'co2_exchange_delta': 'overwrite[float]',
        }

    def _kinetics(self, C_O2):
        return monod_kinetics(C_O2, self._X, self.config)

    def _exchange_fluxes(self, k):
        """Molar specific exchange fluxes (mmol/(gDW·h)) per the cell-side contract.

        Converts the mass-specific Monod ``q_o2`` (g_O2/(gDW·h)) to molar via the
        O2 molecular weight. CO2 production is ``RQ · (molar O2 uptake)`` because
        the respiratory quotient is itself a molar CO2/O2 ratio. Sign convention:
        negative = uptake (O2), positive = evolution (CO2).
        """
        o2_molar = k['q_o2'] / SPECIES_DATA['O2']['MW'] * 1000.0  # mmol/(gDW·h)
        return {
            'OXYGEN-MOLECULE[p]': -o2_molar,
            'CARBON-DIOXIDE[p]': self.config['respiratory_quotient'] * o2_molar,
        }

    def initial_state(self):
        self._X = self.config['initial_biomass_gL']
        k = self._kinetics(self.config['initial_do_mgL'] if 'initial_do_mgL'
                           in self.config else 8.0)
        return {
            'biomass': 0.0,       # delta — store init comes from the composite
            'dissolved_o2': 0.0,
            'dissolved_co2': 0.0,
            'cell_mass': float(self._X),
            'growth_rate': k['mu'],
            'external_exchange_fluxes': self._exchange_fluxes(k),
            'o2_exchange_delta': 0.0,
            'co2_exchange_delta': 0.0,
        }

    def update(self, state, interval):
        if self._X is None:
            self.initial_state()

        C_O2 = state.get('dissolved_o2', 0.0)
        k = self._kinetics(C_O2)

        # Biomass growth (owned by the cell). Held constant when growth disabled.
        dX = k['mu'] * max(self._X, 0.0) * interval if self.config['growth_enabled'] else 0.0
        self._X = max(self._X + dX, 0.0)

        # Exchange applied to shared dissolved stores as deltas. Cap O2 uptake
        # at the O2 actually present so the explicit coupling step can't drive
        # dissolved O2 negative; the exact d[O2]/dt = transport − consumption
        # holds whenever O2 is not limiting (the regime bird-02 validates, and
        # the cap binds less tightly as the coupling interval shrinks — bird-03).
        d_o2 = -min(k['OUR'] * interval, max(C_O2, 0.0))  # consumption (negative)
        d_co2 = k['CER'] * interval                       # evolution (positive)

        return {
            'biomass': dX,
            'dissolved_o2': d_o2,
            'dissolved_co2': d_co2,
            'cell_mass': float(self._X),
            'growth_rate': k['mu'],
            'external_exchange_fluxes': self._exchange_fluxes(k),
            'o2_exchange_delta': d_o2,
            'co2_exchange_delta': d_co2,
        }


class BiRDTransportProcess(Process):
    """Transport-only reactor: emits ONLY the gas-liquid transport contribution.

    Biomass, substrate, and dissolved-gas concentrations are read-only INPUTS;
    the process owns no biomass equation and no substrate consumption. It emits
    the O2/CO2 transport deltas (kLa·(C*−C)) as additive `float` updates to the
    shared dissolved stores, so it composes with a cell engine that supplies the
    consumption side. It NEVER writes the biomass store (biomass is not an
    output), and emits zero glucose transport (no gas-phase glucose).

    This is the REACTOR side of the cell-side interface contract.
    """

    config_schema = {
        'reactor_type': {'_type': 'string', '_default': 'bubble_column'},
        'volume_L': {'_type': 'float', '_default': 20.0},
        'diameter_m': {'_type': 'float', '_default': 0.2},
        'liquid_height_m': {'_type': 'float', '_default': 0.5},
        'gas_flow_rate_Lpm': {'_type': 'float', '_default': 1.0},
        'temperature_K': {'_type': 'float', '_default': 298.15},
        'pressure_atm': {'_type': 'float', '_default': 1.0},
        'o2_fraction_inlet': {'_type': 'float', '_default': 0.21},
        'co2_fraction_inlet': {'_type': 'float', '_default': 0.0004},
        'mean_bubble_diameter_mm': {'_type': 'float', '_default': 3.0},
        'impeller_power_W': {'_type': 'float', '_default': 0.0},
        # 'auto' resolves by geometry (vant_riet for stirred_tank, higbie
        # otherwise); 'higbie' | 'vant_riet' force a specific correlation.
        'kla_correlation': {'_type': 'string', '_default': 'auto'},
    }

    def inputs(self):
        # Biomass / substrate are read-only inputs (never written back).
        return {
            'dissolved_o2': 'float',
            'dissolved_co2': 'float',
            'biomass': 'float',
            'glucose': 'float',
            'gas_flow_rate_Lpm': 'float',
        }

    def outputs(self):
        return {
            # Additive transport deltas to the shared dissolved stores.
            'dissolved_o2': 'float',
            'dissolved_co2': 'float',
            # Diagnostics (overwrite readouts). NOTE: no `biomass`, no `glucose`
            # output — biomass is read-only and glucose has no gas transport.
            'o2_transport_delta': 'overwrite[float]',
            'co2_transport_delta': 'overwrite[float]',
            'glucose_transport': 'overwrite[float]',
            'biomass_transport_delta': 'overwrite[float]',
            'kla_o2': 'overwrite[float]',
            'kla_co2': 'overwrite[float]',
            'o2_saturation': 'overwrite[float]',
            'co2_saturation': 'overwrite[float]',
            'gas_holdup': 'overwrite[float]',
            'superficial_gas_velocity': 'overwrite[float]',
        }

    def _transport(self, C_O2, C_CO2, gas_flow):
        t = compute_transport_state(self.config, gas_flow)
        d_o2_rate = o2_transport_rate(t['kla_o2'], t['cstar_o2'], C_O2)
        d_co2_rate = co2_transport_rate(t['kla_co2'], t['cstar_co2'], C_CO2)
        return t, d_o2_rate, d_co2_rate

    def initial_state(self):
        t, d_o2_rate, d_co2_rate = self._transport(
            8.0, 0.5, self.config['gas_flow_rate_Lpm'])
        return {
            'dissolved_o2': 0.0,
            'dissolved_co2': 0.0,
            'o2_transport_delta': 0.0,
            'co2_transport_delta': 0.0,
            'glucose_transport': 0.0,
            'biomass_transport_delta': 0.0,
            'kla_o2': t['kla_o2'],
            'kla_co2': t['kla_co2'],
            'o2_saturation': t['cstar_o2'],
            'co2_saturation': t['cstar_co2'],
            'gas_holdup': t['alpha_gas'],
            'superficial_gas_velocity': t['Ug'],
        }

    def update(self, state, interval):
        C_O2 = state.get('dissolved_o2', 0.0)
        C_CO2 = state.get('dissolved_co2', 0.0)
        gas_flow = state.get('gas_flow_rate_Lpm', self.config['gas_flow_rate_Lpm'])

        t, d_o2_rate, d_co2_rate = self._transport(C_O2, C_CO2, gas_flow)
        d_o2 = d_o2_rate * interval
        d_co2 = d_co2_rate * interval

        return {
            'dissolved_o2': d_o2,
            'dissolved_co2': d_co2,
            'o2_transport_delta': d_o2,
            'co2_transport_delta': d_co2,
            'glucose_transport': 0.0,
            'biomass_transport_delta': 0.0,
            'kla_o2': t['kla_o2'],
            'kla_co2': t['kla_co2'],
            'o2_saturation': t['cstar_o2'],
            'co2_saturation': t['cstar_co2'],
            'gas_holdup': t['alpha_gas'],
            'superficial_gas_velocity': t['Ug'],
        }


class BiRDColumn1DProcess(Process):
    """1D depth-resolved (axial) bubble-column reactor.

    Where BiRDReactorProcess is 0D (well-mixed, one scalar per species), this
    process discretizes the liquid height into ``n_nodes`` stacked control
    volumes (index 0 = bottom / sparger, n_nodes-1 = top / free surface) and
    resolves the dissolved-species and biomass profiles DOWN the column,
    evolving in time. That gives genuine depth x time fields to visualize as
    heatmaps — the spatial state a 0D model cannot express.

    The physics reuses the SAME shared correlations as the 0D reactor
    (``transport.compute_transport_state`` for kLa / gas holdup, Henry's-law
    ``saturation_concentration`` for C*), so a tall column reduces to the 0D
    reactor in the well-mixed limit. Three real effects create the axial
    gradient:

      1. **Hydrostatic pressure.** P(z) = P_head + rho*g*(H - z); deeper liquid
         is at higher pressure, so O2 solubility C* is highest at the sparger.
      2. **Gas-phase O2 depletion.** Gas enters at the bottom and gives up O2 to
         the liquid as it rises (a plug-flow gas balance), so the gas-phase O2
         mole fraction — and thus C* — falls toward the top.
      3. **Axial liquid dispersion** (Fickian, ``axial_dispersion_m2_s``)
         couples neighbouring nodes; consumption (Monod OUR) draws each node
         down between transfer and mixing.

    Per-node liquid balance (mg/(L.h)):
        dC_O2/dt  = kLa*(C*(z) - C_O2)  - OUR(z)  + D_ax d2C_O2/dz2
        dC_CO2/dt = CER(z) - kLa*(C_CO2 - C*_CO2) + D_ax d2C_CO2/dz2
        dX/dt     = mu(z)*X               + D_ax d2X/dz2
    with no-flux (Neumann) boundaries at the sparger and free surface. Time
    units: hours; concentrations mg/L (gas species) or g/L (biomass); depth m.
    """

    R_GAS = 8.314          # J/(mol.K)
    P_ATM_PA = 101325.0    # Pa per atm

    config_schema = {
        # Geometry / discretization
        'reactor_type': {'_type': 'string', '_default': 'bubble_column'},
        'n_nodes': {'_type': 'integer', '_default': 20},
        'volume_L': {'_type': 'float', '_default': 2000.0},
        'diameter_m': {'_type': 'float', '_default': 0.6},
        'liquid_height_m': {'_type': 'float', '_default': 6.0},
        'axial_dispersion_m2_s': {'_type': 'float', '_default': 0.03},
        # Operating conditions
        'gas_flow_rate_Lpm': {'_type': 'float', '_default': 150.0},
        'temperature_K': {'_type': 'float', '_default': 303.15},
        'pressure_atm': {'_type': 'float', '_default': 1.0},
        'o2_fraction_inlet': {'_type': 'float', '_default': 0.21},
        'co2_fraction_inlet': {'_type': 'float', '_default': 0.0004},
        'mean_bubble_diameter_mm': {'_type': 'float', '_default': 3.0},
        # Microbial kinetics (shared with the 0D reactor)
        'initial_biomass_gL': {'_type': 'float', '_default': 1.0},
        'initial_do_mgL': {'_type': 'float', '_default': 8.0},
        'initial_dco2_mgL': {'_type': 'float', '_default': 0.5},
        'max_growth_rate_per_h': {'_type': 'float', '_default': 0.05},
        'ks_oxygen_mgL': {'_type': 'float', '_default': 0.2},
        'yield_biomass_o2': {'_type': 'float', '_default': 1.2},
        'maintenance_coeff_per_h': {'_type': 'float', '_default': 0.03},
        'respiratory_quotient': {'_type': 'float', '_default': 1.0},
        'impeller_power_W': {'_type': 'float', '_default': 0.0},
        'kla_correlation': {'_type': 'string', '_default': 'auto'},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self._O2 = None    # np.ndarray[n_nodes], dissolved O2 (mg/L), index 0 = bottom
        self._CO2 = None
        self._X = None
        self._last = {}     # last derived profiles (for emit)

    # ports ------------------------------------------------------------------
    def inputs(self):
        return {'gas_flow_rate_Lpm': 'float'}

    def outputs(self):
        return {
            # depth-resolved profiles (index 0 = bottom); overwrite each tick so
            # the emitter accumulates a [time, depth] field for the heatmap.
            'dissolved_o2_profile': 'overwrite[list[float]]',
            'dissolved_co2_profile': 'overwrite[list[float]]',
            'biomass_profile': 'overwrite[list[float]]',
            'o2_saturation_profile': 'overwrite[list[float]]',
            'depths_m': 'overwrite[list[float]]',
            # volume-averaged scalars (parity with the 0D reactor + summary panels)
            'dissolved_o2': 'overwrite[float]',
            'dissolved_co2': 'overwrite[float]',
            'biomass': 'overwrite[float]',
            'do_bottom': 'overwrite[float]',
            'do_top': 'overwrite[float]',
            'do_gradient': 'overwrite[float]',
            'kla_o2': 'overwrite[float]',
            'gas_holdup': 'overwrite[float]',
            'o2_uptake_rate': 'overwrite[float]',
            'specific_growth_rate': 'overwrite[float]',
            'superficial_gas_velocity': 'overwrite[float]',
        }

    # geometry helpers -------------------------------------------------------
    def _node_centers(self):
        cfg = self.config
        n = int(cfg['n_nodes'])
        H = cfg['liquid_height_m']
        dz = H / n
        # index 0 = bottom (z small), index n-1 = top (z ~ H)
        return np.array([(i + 0.5) * dz for i in range(n)]), dz

    def _hydrostatic_pressure_atm(self, z_centers):
        """P(z) in atm: head pressure + rho*g*(depth below free surface)."""
        cfg = self.config
        H = cfg['liquid_height_m']
        depth_below = H - z_centers  # m below free surface (bottom = deepest)
        return cfg['pressure_atm'] + RHO_LIQ * G * depth_below / self.P_ATM_PA

    def _gas_molar_flow_per_h(self, gas_flow_Lpm):
        """Total inlet gas molar flow (mol/h) from ideal gas at inlet P,T."""
        cfg = self.config
        V_m3_per_h = gas_flow_Lpm * 60.0 / 1000.0
        P_pa = cfg['pressure_atm'] * self.P_ATM_PA
        return P_pa * V_m3_per_h / (self.R_GAS * cfg['temperature_K'])

    def _saturation_profiles(self, z_centers, gas_flow_Lpm):
        """C*_O2(z), C*_CO2(z) (mg/L) from hydrostatic P(z) + plug-flow gas-O2 depletion.

        Marches the gas stream bottom -> top: at each node the O2 transferred to
        the liquid (kLa*(C*-C)) is removed from the rising gas, lowering the
        gas-phase O2 mole fraction (and thus C*) at nodes above.
        """
        cfg = self.config
        T = cfg['temperature_K']
        P_z = self._hydrostatic_pressure_atm(z_centers)
        n = len(z_centers)
        V_node_L = cfg['volume_L'] / n
        t = compute_transport_state(cfg, gas_flow_Lpm)
        kla_o2 = t['kla_o2']

        n_gas_total = self._gas_molar_flow_per_h(gas_flow_Lpm)   # mol/h
        n_O2 = n_gas_total * cfg['o2_fraction_inlet']            # mol/h entering at bottom
        MW_O2 = SPECIES_DATA['O2']['MW']                         # g/mol

        cstar_o2 = np.zeros(n)
        for i in range(n):  # bottom (0) -> top (n-1)
            y_O2 = max(n_O2 / max(n_gas_total, 1e-12), 1e-6)
            cstar_o2[i] = saturation_concentration('O2', T, P_z[i] * y_O2)
            # O2 given up to the liquid at this node (mol/h), removed from gas stream
            transfer_mgLh = max(kla_o2 * (cstar_o2[i] - self._O2[i]), 0.0)
            transfer_mol_h = transfer_mgLh * V_node_L / 1000.0 / MW_O2
            n_O2 = max(n_O2 - transfer_mol_h, 1e-9)

        cstar_co2 = np.array([
            saturation_concentration('CO2', T, P_z[i] * cfg['co2_fraction_inlet'])
            for i in range(n)
        ])
        return cstar_o2, cstar_co2, t

    # lifecycle --------------------------------------------------------------
    def initial_state(self):
        cfg = self.config
        n = int(cfg['n_nodes'])
        self._O2 = np.full(n, float(cfg['initial_do_mgL']))
        self._CO2 = np.full(n, float(cfg['initial_dco2_mgL']))
        self._X = np.full(n, float(cfg['initial_biomass_gL']))
        z, _ = self._node_centers()
        return self._emit(z, *self._saturation_profiles(z, cfg['gas_flow_rate_Lpm']))

    def _emit(self, z, cstar_o2, cstar_co2, t):
        cfg = self.config
        O2p = np.maximum(self._O2, 0.0)
        mu = cfg['max_growth_rate_per_h'] * O2p / (cfg['ks_oxygen_mgL'] + O2p)
        our = (mu / max(cfg['yield_biomass_o2'], 1e-12) + cfg['maintenance_coeff_per_h']) \
            * np.maximum(self._X, 0.0) * 1000.0
        return {
            'dissolved_o2_profile': [float(v) for v in self._O2],
            'dissolved_co2_profile': [float(v) for v in self._CO2],
            'biomass_profile': [float(v) for v in self._X],
            'o2_saturation_profile': [float(v) for v in cstar_o2],
            'depths_m': [float(v) for v in z],
            'dissolved_o2': float(self._O2.mean()),
            'dissolved_co2': float(self._CO2.mean()),
            'biomass': float(self._X.mean()),
            'do_bottom': float(self._O2[0]),
            'do_top': float(self._O2[-1]),
            'do_gradient': float(self._O2[0] - self._O2[-1]),
            'kla_o2': float(t['kla_o2']),
            'gas_holdup': float(t['alpha_gas']),
            'o2_uptake_rate': float(our.mean()),
            'specific_growth_rate': float(mu.mean()),
            'superficial_gas_velocity': float(t['Ug']),
        }

    def update(self, state, interval):
        if self._O2 is None:
            self.initial_state()
        cfg = self.config
        gas_flow = state.get('gas_flow_rate_Lpm', cfg['gas_flow_rate_Lpm'])
        z, dz = self._node_centers()
        D_ax_h = cfg['axial_dispersion_m2_s'] * 3600.0   # m^2/s -> m^2/h

        # Explicit sub-stepping: satisfy BOTH the diffusion CFL limit
        # dt < dz^2/(2 D_ax) and the transport-relaxation limit dt < 1/kLa, so a
        # stiff kLa or fine grid can't overshoot. (Deep O2-limited regimes can
        # still zero-clamp; the demo composites stay in the aerobic regime.)
        _t0 = compute_transport_state(cfg, gas_flow)
        dt_diff = 0.4 * dz * dz / max(D_ax_h, 1e-12)
        dt_react = 0.5 / max(_t0['kla_o2'], 1e-9)
        # Cap the substep count so an extreme dispersion / fine grid can't make
        # the explicit loop explode. At the cap the profile just approaches the
        # well-mixed limit (large D_ax), which is the physically correct end.
        MAX_SUBSTEPS = 3000
        n_sub = max(1, int(math.ceil(interval / min(dt_diff, dt_react, interval))))
        n_sub = min(n_sub, MAX_SUBSTEPS)
        dt = interval / n_sub
        # Clamp the effective diffusion to the explicit-stability limit for this
        # dt, so an arbitrarily large axial_dispersion (or the substep cap above)
        # can never make the explicit Laplacian blow up. Physically this means
        # "mix at most as fast as the grid+step can resolve" — a larger D_ax then
        # saturates at the well-mixed limit instead of producing NaNs.
        D_eff = min(D_ax_h, 0.5 * dz * dz / dt)

        # Vectorized Monod coefficients (mg/(L.h)) — pure numpy, no per-node call.
        mg = cfg['max_growth_rate_per_h']
        ks = cfg['ks_oxygen_mgL']
        inv_yield = 1.0 / max(cfg['yield_biomass_o2'], 1e-12)
        m_coeff = cfg['maintenance_coeff_per_h']
        rq_factor = cfg['respiratory_quotient'] * (44.01 / 32.0)

        def lap(a):
            pad = np.concatenate(([a[0]], a, [a[-1]]))
            return (pad[2:] - 2.0 * pad[1:-1] + pad[:-2]) / (dz * dz)

        cstar_o2 = cstar_co2 = None
        t = None
        for _ in range(n_sub):
            cstar_o2, cstar_co2, t = self._saturation_profiles(z, gas_flow)
            O2, CO2, X = self._O2, self._CO2, self._X
            O2p = np.maximum(O2, 0.0)
            mu = mg * O2p / (ks + O2p)                       # 1/h
            our = (mu * inv_yield + m_coeff) * np.maximum(X, 0.0) * 1000.0  # mg/(L.h)
            cer = rq_factor * our

            dO2 = t['kla_o2'] * (cstar_o2 - O2) - our + D_eff * lap(O2)
            dCO2 = cer - t['kla_co2'] * (CO2 - cstar_co2) + D_eff * lap(CO2)
            dX = mu * X + D_eff * lap(X)

            self._O2 = np.maximum(O2 + dO2 * dt, 0.0)
            self._CO2 = np.maximum(CO2 + dCO2 * dt, 0.0)
            self._X = np.maximum(X + dX * dt, 0.0)

        return self._emit(z, cstar_o2, cstar_co2, t)
