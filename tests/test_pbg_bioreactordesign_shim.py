"""The pbg_bioreactordesign import name stays working (deprecated) after the rename."""
import sys
import warnings


def test_pbg_bioreactordesign_still_imports_and_warns():
    # Drop any cached import so the deprecation warning re-fires inside the
    # catch_warnings block (another test in the suite may have imported the
    # shim first, which would otherwise suppress the warning on re-import).
    for mod in [m for m in sys.modules if m == "pbg_bioreactordesign" or m.startswith("pbg_bioreactordesign.")]:
        del sys.modules[mod]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        import pbg_bioreactordesign  # noqa: F401
    assert any(issubclass(x.category, DeprecationWarning) for x in w)


def test_pbg_bioreactordesign_submodule_redirects_to_viva_bioreactordesign():
    import viva_bioreactordesign.transport as real
    import pbg_bioreactordesign.transport as shimmed
    assert shimmed is real            # meta-path finder aliases to the real module
