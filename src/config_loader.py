import os
import yaml
from dataclasses import dataclass
from typing import Dict, Any


# -------------------------------------------------
# Utility
# -------------------------------------------------

def _load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing config file: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _project_root() -> str:
    """
    Returns project root assuming this file lives in src/
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# -------------------------------------------------
# Config Containers
# -------------------------------------------------

@dataclass(frozen=True)
class FullConfig:
    project: Dict[str, Any]
    regulatory: Dict[str, Any]
    tariffs: Dict[str, Any]
    bess: Dict[str, Any]
    finance: Dict[str, Any]
    solver: Dict[str, Any]


# -------------------------------------------------
# Validation
# -------------------------------------------------

def _validate(config: FullConfig):

    # 1. Resolution check
    if config.project["simulation"]["resolution"] != "hourly":
        raise ValueError("Only hourly resolution supported.")

    # 2. Annual constraint validity
    allowed = ["none", "replacement", "cuf"]
    if config.solver["solver"]["annual_constraint"]["type"] not in allowed:
        raise ValueError("Invalid annual constraint type.")

    # 3. Container size > 0
    if config.bess["bess"]["container"]["size_mwh"] <= 0:
        raise ValueError("BESS container size must be positive.")

    # 4. AC:DC ratio > 0
    if config.finance["capex"]["solar"]["ac_dc_ratio"] <= 0:
        raise ValueError("AC:DC ratio must be positive.")

    print("✓ Config validation successful.")


# -------------------------------------------------
# Public Loader
# -------------------------------------------------

def load_config() -> FullConfig:

    root = _project_root()
    config_dir = os.path.join(root, "configs")

    project = _load_yaml(os.path.join(config_dir, "project.yaml"))
    regulatory = _load_yaml(os.path.join(config_dir, "regulatory.yaml"))
    tariffs = _load_yaml(os.path.join(config_dir, "tariffs.yaml"))
    bess = _load_yaml(os.path.join(config_dir, "bess.yaml"))
    finance = _load_yaml(os.path.join(config_dir, "finance.yaml"))
    solver = _load_yaml(os.path.join(config_dir, "solver.yaml"))

    config = FullConfig(
        project=project,
        regulatory=regulatory,
        tariffs=tariffs,
        bess=bess,
        finance=finance,
        solver=solver
    )

    _validate(config)

    return config