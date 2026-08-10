"""tests/style_consistency_tests/test_packaging_ships_env_assets.py.

Guards the packaging rule that ships every environment-definition asset in the wheel.

``standalone/`` and ``shared_envs/`` directories hold non-Python files that tool env
builds read at runtime: ``setup.sh``, ``requirements.txt``, ``python_version.txt``,
``env_vars.txt``, container recipes. Package discovery only picks up ``.py`` files;
everything else ships only if its extension appears in
``[tool.setuptools.package-data]``. An asset missing from that list is present in a
clone or editable install and absent from a ``pip install``, so the tool builds in
development and fails for users on the same commit.
"""

from __future__ import annotations

import fnmatch
import subprocess
from pathlib import Path

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # requires-python allows 3.10, where tomllib does not exist
    import tomli as tomllib  # type: ignore[no-redef]

_REPO_ROOT = Path(__file__).parent.parent.parent
_PYPROJECT = _REPO_ROOT / "pyproject.toml"


def _package_data_patterns() -> list[str]:
    """Return the ``[tool.setuptools.package-data]`` globs applied to every package."""
    with _PYPROJECT.open("rb") as fh:
        data = tomllib.load(fh)
    return data["tool"]["setuptools"]["package-data"]["*"]


def _tracked_env_assets() -> list[str]:
    """Return tracked non-Python files under any ``standalone/`` dir or ``shared_envs/``."""
    tracked = subprocess.run(
        ["git", "ls-files", "proto_tools"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return [
        path
        for path in tracked
        if not path.endswith(".py") and ("/standalone/" in path or path.startswith("proto_tools/shared_envs/"))
    ]


def test_every_env_asset_matches_a_package_data_glob() -> None:
    """Every shipped env-definition file must match a package-data pattern."""
    patterns = _package_data_patterns()
    missing = [
        path
        for path in _tracked_env_assets()
        if not any(fnmatch.fnmatch(Path(path).name, pattern) for pattern in patterns)
    ]
    assert not missing, (
        "these env-definition files would be missing from a wheel install; add their "
        "extension to [tool.setuptools.package-data]:\n" + "\n".join(f"  {p}" for p in missing)
    )


def test_env_assets_are_discovered() -> None:
    """A discovery bug that found no assets would make the check above pass vacuously."""
    assert len(_tracked_env_assets()) > 100
