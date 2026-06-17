import tomllib
import types
import importlib.util
from pathlib import Path

_PYTEST_PARALLEL_PATH = Path("scripts/pytest_parallel.py").resolve()
_PYTEST_PARALLEL_SPEC = importlib.util.spec_from_file_location(
    "pytest_parallel_script",
    _PYTEST_PARALLEL_PATH,
)
assert _PYTEST_PARALLEL_SPEC is not None
pytest_parallel = importlib.util.module_from_spec(_PYTEST_PARALLEL_SPEC)
assert _PYTEST_PARALLEL_SPEC.loader is not None
_PYTEST_PARALLEL_SPEC.loader.exec_module(pytest_parallel)


def test_pytest_config_supports_parallel_test_runs_without_forcing_default_pytest():
    config = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    test_dependencies = config["project"]["optional-dependencies"]["test"]
    addopts = config["tool"]["pytest"]["ini_options"]["addopts"]

    assert any(dependency.startswith("pytest-xdist") for dependency in test_dependencies)
    assert "-n" not in addopts


def test_parallel_test_runner_uses_repo_local_temp_root(monkeypatch):
    calls = []
    monkeypatch.setattr(pytest_parallel, "_run_pytest", lambda args: calls.append(args) or 0)

    result = pytest_parallel.main(["tests/test_pytest_config.py"])

    assert result == 0
    args = calls[0]
    assert args[:3] == ["-n", "auto", "-q"]
    assert args[-1] == "tests/test_pytest_config.py"
    basetemp_arg = next(argument for argument in args if argument.startswith("--basetemp="))
    temp_root = Path(basetemp_arg.split("=", 1)[1])
    assert temp_root.parent == Path("tests/.tmp").resolve()
    assert temp_root.name.startswith("pytest-parallel-")


def test_parallel_test_runner_restores_path_mkdir_after_pytest(monkeypatch):
    original_mkdir = pytest_parallel.Path.mkdir
    monkeypatch.setitem(pytest_parallel.sys.modules, "pytest", types.SimpleNamespace(main=lambda args: 0))

    result = pytest_parallel._run_pytest([])

    assert result == 0
    assert pytest_parallel.Path.mkdir is original_mkdir
