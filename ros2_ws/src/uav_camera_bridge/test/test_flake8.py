"""Check Python style."""

from ament_flake8.main import main_with_errors

import pytest


@pytest.mark.flake8
@pytest.mark.linter
def test_flake8():
    """Run flake8."""
    rc, errors = main_with_errors(argv=["--config", "setup.cfg"])
    message = "Found %d code style errors:\n" % len(errors)
    assert rc == 0, message + "\n".join(errors)
