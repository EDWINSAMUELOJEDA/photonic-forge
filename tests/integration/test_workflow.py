"""Integration tests for end-to-end workflows."""

import pytest


@pytest.mark.integration
class TestWorkflow:
    """Placeholder integration tests for complete workflows."""

    def test_import_all_modules(self):
        """Test that all main modules can be imported together."""
        import photonic_forge
        import photonic_forge.core
        import photonic_forge.solvers
        import photonic_forge.optimize
        import photonic_forge.neural_models
        import photonic_forge.agent
        import photonic_forge.pdk
        import photonic_forge.ui

        assert photonic_forge.__version__ == "0.1.0"
