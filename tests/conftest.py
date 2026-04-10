from __future__ import annotations

from pathlib import Path

import pytest

from .helpers import write_small_benchmark_config


@pytest.fixture
def benchmark_config_path(tmp_path: Path) -> Path:
    return write_small_benchmark_config(tmp_path / "benchmark_config.json")
