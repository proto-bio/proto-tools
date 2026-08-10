"""tests/tool_infra_tests/test_env_report_confirmation.py.

Tests for the ``--env-report`` deletion guard.

``--env-report`` removes built tool environments and rebuilds them, which costs
hours and re-downloads gigabytes. The guard makes that destructive step
explicit. These tests exercise it directly rather than through a real run, so
no environment is ever at risk.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import _confirm_env_deletion


class _Config:
    """Minimal stand-in for the pytest config object the guard consumes."""

    def __init__(self, yes: bool) -> None:
        self._yes = yes
        self.pluginmanager = _PluginManager()

    def getoption(self, name: str) -> bool:
        assert name == "--yes"
        return self._yes


class _PluginManager:
    def getplugin(self, name: str) -> None:
        """No capture manager under test; the guard must tolerate its absence."""
        return


TARGETS = [Path("/fake/envs/chai1_env"), Path("/fake/envs/germinal_env")]


def test_yes_flag_bypasses_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    """-Y/--yes must not prompt, so CI and unattended runs work."""
    monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("should not prompt"))
    _confirm_env_deletion(_Config(yes=True), TARGETS, full_root=None)


def test_non_interactive_without_yes_aborts(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-tty session must abort rather than guess, so a cron run cannot wipe envs."""
    monkeypatch.setattr("sys.stdin", None)
    with pytest.raises(Exception, match="stdin is not interactive") as exc:
        _confirm_env_deletion(_Config(yes=False), TARGETS, full_root=None)
    assert exc.type.__name__ == "Exit"


def test_typing_yes_proceeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.stdin", _Tty())
    monkeypatch.setattr("builtins.input", lambda *a: "yes")
    _confirm_env_deletion(_Config(yes=False), TARGETS, full_root=None)


@pytest.mark.parametrize("answer", ["", "n", "no", "Y", "yes please"])
def test_anything_but_yes_aborts(monkeypatch: pytest.MonkeyPatch, answer: str) -> None:
    """Only an exact "yes" proceeds; a stray keypress must not delete environments."""
    monkeypatch.setattr("sys.stdin", _Tty())
    monkeypatch.setattr("builtins.input", lambda *a: answer)
    with pytest.raises(Exception, match="aborted") as exc:
        _confirm_env_deletion(_Config(yes=False), TARGETS, full_root=Path("/fake/envs"))
    assert exc.type.__name__ == "Exit"


class _Tty:
    def isatty(self) -> bool:
        return True
