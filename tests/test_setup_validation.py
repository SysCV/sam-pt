"""Validation tests to verify the testing infrastructure is properly set up."""
import os
import sys
from pathlib import Path

import pytest


class TestInfrastructureValidation:
    """Validate that the testing infrastructure is properly configured."""

    def test_pytest_installed(self):
        """Verify pytest is available."""
        assert "pytest" in sys.modules or __name__ == "pytest"

    def test_coverage_configured(self):
        """Verify coverage is properly configured."""
        try:
            import coverage
            assert coverage is not None
        except ImportError:
            pytest.fail("Coverage module not installed")

    def test_project_structure(self):
        """Verify the project structure is correct."""
        project_root = Path(__file__).parent.parent
        
        # Check essential directories exist
        assert project_root.exists()
        assert (project_root / "sam_pt").exists()
        assert (project_root / "demo").exists()
        assert (project_root / "tests").exists()
        assert (project_root / "tests" / "unit").exists()
        assert (project_root / "tests" / "integration").exists()

    def test_pyproject_toml_exists(self):
        """Verify pyproject.toml exists and contains required sections."""
        project_root = Path(__file__).parent.parent
        pyproject_path = project_root / "pyproject.toml"
        
        assert pyproject_path.exists(), "pyproject.toml not found"
        
        content = pyproject_path.read_text()
        assert "[tool.poetry]" in content
        assert "[tool.pytest.ini_options]" in content
        assert "[tool.coverage.run]" in content

    def test_conftest_fixtures(self, temp_dir, mock_config):
        """Test that conftest fixtures are available."""
        assert temp_dir.exists()
        assert isinstance(mock_config, dict)
        assert "model" in mock_config

    @pytest.mark.unit
    def test_unit_marker(self):
        """Test that unit test marker works."""
        assert True

    @pytest.mark.integration
    def test_integration_marker(self):
        """Test that integration test marker works."""
        assert True

    @pytest.mark.slow
    def test_slow_marker(self):
        """Test that slow test marker works."""
        assert True

    def test_mock_functionality(self, mock_torch_model):
        """Test that mocking functionality works."""
        assert mock_torch_model is not None
        mock_torch_model.eval()
        mock_torch_model.eval.assert_called_once()

    def test_sample_fixtures(self, sample_image, sample_points, sample_masks):
        """Test that sample data fixtures work correctly."""
        assert sample_image.exists()
        assert len(sample_points) == 3
        assert len(sample_points[0]) == 2
        assert len(sample_masks) == 3

    def test_env_setup(self, env_setup):
        """Test that environment setup fixture works."""
        assert os.environ.get("TEST_MODE") == "1"
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"


class TestPoetryScripts:
    """Validate Poetry scripts are configured correctly."""

    def test_poetry_scripts_configured(self):
        """Verify Poetry test scripts are in pyproject.toml."""
        project_root = Path(__file__).parent.parent
        pyproject_path = project_root / "pyproject.toml"
        
        content = pyproject_path.read_text()
        assert 'test = "pytest:main"' in content
        assert 'tests = "pytest:main"' in content