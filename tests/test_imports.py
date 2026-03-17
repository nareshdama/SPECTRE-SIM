import importlib
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def test_external_dependencies_import() -> None:
    modules = ["numpy", "scipy", "matplotlib", "yaml", "pandas", "pytest", "tqdm"]

    for module_name in modules:
        assert importlib.import_module(module_name) is not None


def test_internal_packages_import() -> None:
    modules = ["engagement", "guidance", "estimator", "attacker", "monitor", "utils"]

    for module_name in modules:
        assert importlib.import_module(module_name) is not None
