"""Shared pytest fixtures for the SAM-PT project."""
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any, Generator
from unittest.mock import MagicMock

import pytest
# PIL Image import removed - would be imported when PIL is available


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Provide a mock configuration dictionary."""
    return {
        "model": {
            "name": "test_model",
            "type": "sam_pt",
            "checkpoint": None,
        },
        "data": {
            "batch_size": 2,
            "num_workers": 0,
            "dataset": "test_dataset",
        },
        "training": {
            "epochs": 1,
            "learning_rate": 0.001,
            "optimizer": "adam",
        },
        "logging": {
            "level": "INFO",
            "dir": "./logs",
        },
    }


@pytest.fixture
def sample_image(temp_dir: Path) -> Path:
    """Create a sample image for testing."""
    img_path = temp_dir / "test_image.png"
    # Create a dummy file to simulate an image
    img_path.write_text("mock image data")
    return img_path


@pytest.fixture
def sample_video_frames(temp_dir: Path) -> Path:
    """Create sample video frames for testing."""
    frames_dir = temp_dir / "frames"
    frames_dir.mkdir()
    
    for i in range(5):
        # Create dummy files to simulate video frames
        frame_path = frames_dir / f"frame_{i:03d}.png"
        frame_path.write_text(f"mock frame {i} data")
    
    return frames_dir


@pytest.fixture
def mock_torch_model() -> MagicMock:
    """Create a mock PyTorch model."""
    model = MagicMock()
    model.eval = MagicMock(return_value=model)
    model.train = MagicMock(return_value=model)
    model.parameters = MagicMock(return_value=[MagicMock()])
    model.state_dict = MagicMock(return_value={"test": MagicMock()})
    return model


@pytest.fixture
def sample_points() -> list:
    """Generate sample query points for tracking."""
    return [
        [100.0, 150.0],
        [320.0, 240.0],
        [500.0, 400.0],
    ]


@pytest.fixture
def sample_masks() -> list:
    """Generate sample segmentation masks."""
    # Return a simple list representation of masks instead of numpy array
    return [
        {"x": 100, "y": 100, "width": 100, "height": 100},
        {"x": 300, "y": 200, "width": 100, "height": 100},
        {"x": 450, "y": 350, "width": 100, "height": 100},
    ]


@pytest.fixture
def mock_dataset() -> MagicMock:
    """Create a mock dataset."""
    dataset = MagicMock()
    dataset.__len__ = MagicMock(return_value=10)
    dataset.__getitem__ = MagicMock(return_value={
        "image": MagicMock(shape=(3, 480, 640)),
        "mask": MagicMock(shape=(1, 480, 640)),
        "points": [[100.0, 150.0]],
    })
    return dataset


@pytest.fixture
def mock_checkpoint(temp_dir: Path) -> Path:
    """Create a mock checkpoint file."""
    checkpoint_path = temp_dir / "model_checkpoint.pth"
    # Create a dummy file to simulate checkpoint
    checkpoint_path.write_text("mock checkpoint data")
    return checkpoint_path


@pytest.fixture
def env_setup(monkeypatch) -> None:
    """Set up test environment variables."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")


@pytest.fixture(autouse=True)
def cleanup_matplotlib():
    """Clean up matplotlib figures after each test."""
    try:
        import matplotlib.pyplot as plt
        yield
        plt.close('all')
    except ImportError:
        # matplotlib not installed yet, skip cleanup
        yield


@pytest.fixture
def mock_wandb(monkeypatch) -> MagicMock:
    """Mock wandb for tests."""
    mock = MagicMock()
    mock.init = MagicMock()
    mock.log = MagicMock()
    mock.finish = MagicMock()
    monkeypatch.setattr("wandb.init", mock.init)
    monkeypatch.setattr("wandb.log", mock.log)
    monkeypatch.setattr("wandb.finish", mock.finish)
    return mock