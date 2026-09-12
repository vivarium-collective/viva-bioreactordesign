"""Uniform ``build_core`` entry point for viva-bioreactordesign.

Follows the cross-repo convention every viva-/pbg- repo exposes: a single
``build_core(core=None)`` that (1) allocates a fresh process-bigraph core when
``core is None`` (else composes onto the passed-in core), (2) registers THIS
repo's own process/step classes by name so they are first-class, browsable
dashboard Registry entries, and (3) returns the core.

The shared helper ``viva_superpowers.core_compose.register_package_processes``
does the registration: it filters to real ``process_bigraph`` Process/Step
subclasses, and is best-effort + idempotent.
"""
from __future__ import annotations

from process_bigraph import allocate_core

from viva_bioreactordesign.processes import BiRDTransportProcess


def build_core(core=None):
    """Return a core with viva-bioreactordesign's own processes registered.

    Registers every process-bigraph-native Process/Step class defined in the
    ``viva_bioreactordesign`` package (BiRDReactorProcess, MonodCellProcess,
    BiRDTransportProcess, and the BioreactorPlots visualization) so they show up
    as first-class dashboard Registry entries. Additive and idempotent — safe to
    compose onto a core that already carries other repos' registrations.
    """
    if core is None:
        core = allocate_core()

    # Register this repo's own processes/steps by name via the shared helper.
    try:
        from viva_superpowers.core_compose import register_package_processes

        register_package_processes(core, "viva_bioreactordesign")
    except Exception:
        # Best-effort: the shared helper may be unavailable; fall through to the
        # explicit registration below so the primary process is always present.
        pass

    # BiRDTransportProcess is a top-level import here — register it explicitly so
    # it is guaranteed present even if the package scan above is skipped.
    try:
        core.register_link("BiRDTransportProcess", BiRDTransportProcess)
    except Exception:
        pass

    return core


__all__ = ["build_core"]
