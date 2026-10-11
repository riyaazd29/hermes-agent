"""Process teardown behind ``search_files`` must never fail the caller.

The native rg runner drains rg's output, then TERMs the process group through the shared
teardown helper. When the child has already been reaped and no process group was recorded
(the native rg lane's shape), ``os.getpgid`` raises ``ProcessLookupError`` and the helper's
bare re-raise escaped all the way to the tool result — ``{"error": "[Errno 3] No such
process"}`` replacing matches the runner had already drained. Same class: ``os.getpgid``
answering ``PermissionError`` when the pid now belongs to a foreign process. And when a
kill does fail, the drained matches still belong to the caller.

Teardown must be silent-by-construction, the runner must record the group at spawn so the
helper keeps a fallback, and a teardown failure must never replace drained output.
"""
import os
import subprocess
import sys
import types

import pytest

from tools.environments.local import _kill_process_group_posix

pytestmark = pytest.mark.platforms("posix")


@pytest.fixture
def ops(tmp_path):
    from tools.environments.local import LocalEnvironment
    from tools.file_operations import ShellFileOperations
    env = LocalEnvironment(cwd=str(tmp_path))
    return ShellFileOperations(env, cwd=str(tmp_path))


def _throw(exc):
    raise exc


def test_teardown_of_a_reaped_child_without_a_recorded_group_is_silent(monkeypatch):
    """The reproduced case: reaped child, no ``_hermes_pgid`` recorded -> must not raise."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    proc.wait()  # reap: the pid is gone from the process table
    monkeypatch.setattr(
        os, "getpgid",
        lambda pid: _throw(ProcessLookupError(3, "No such process")))
    _kill_process_group_posix(proc)  # must return quietly


def test_teardown_survives_getpgid_permission_error(monkeypatch):
    """A recycled pid owned by a foreign process: signal nothing, raise nothing."""
    proc = types.SimpleNamespace(pid=123456)
    monkeypatch.setattr(
        os, "getpgid",
        lambda pid: _throw(PermissionError(1, "Operation not permitted")))
    _kill_process_group_posix(proc)  # must return quietly


def test_native_rg_runner_records_the_process_group_for_teardown(ops, monkeypatch):
    """The runner must leave the helper a fallback (``_hermes_pgid``) for the reaped case."""
    captured = []
    real_popen = subprocess.Popen

    def spy(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        captured.append(proc)
        return proc

    monkeypatch.setattr("tools.file_operations_search.subprocess.Popen", spy)
    result = ops._run_rg_native(["sh", "-c", "'echo needle'"], 5, timeout=10)
    assert result.exit_code == 0
    (proc,) = captured
    assert isinstance(getattr(proc, "_hermes_pgid", None), int) and proc._hermes_pgid > 0
    assert getattr(proc, "_hermes_pgid_start", None) is not None


def test_teardown_failure_never_replaces_drained_matches(ops, monkeypatch):
    """Bound reached while rg still runs -> teardown runs; a failing teardown must not
    take the drained matches down with it."""
    monkeypatch.setattr(
        "tools.environments.local._kill_process_group_posix",
        lambda proc: _throw(OSError(3, "No such process")))
    argv = ["sh", "-c", "'echo needle-one; echo needle-two; echo needle-three; sleep 30'"]
    result = ops._run_rg_native(argv, 2, timeout=10)
    assert result.exit_code == 0
    assert "needle-one" in result.stdout and "needle-two" in result.stdout, result.stdout
    assert "needle-three" not in result.stdout, result.stdout
