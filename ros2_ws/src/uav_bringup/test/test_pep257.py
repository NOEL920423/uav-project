"""Check Python docstrings."""

from ament_pep257.main import main

import pytest


@pytest.mark.linter
@pytest.mark.pep257
def test_pep257():
    """Run pydocstyle."""
    assert main(argv=["."]) == 0
