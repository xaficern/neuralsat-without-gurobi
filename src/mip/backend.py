from __future__ import annotations

from types import SimpleNamespace


class BackendUnavailableError(RuntimeError):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


_BACKEND_IMPORT_ERROR = None
_gurobi_module = None

try:
    import gurobipy as _candidate_gurobi

    _test_model = _candidate_gurobi.Model("neuralsat_backend_probe")
    _test_model.setParam("OutputFlag", 0)
    _test_model = None
    _gurobi_module = _candidate_gurobi
except Exception as exc:
    _BACKEND_IMPORT_ERROR = str(exc)


class _UnavailableGRB:
    GRB = SimpleNamespace(
        CONTINUOUS="CONTINUOUS",
        BINARY="BINARY",
        MINIMIZE=1,
        MAXIMIZE=-1,
        INFEASIBLE=3,
    )
    GurobiError = BackendUnavailableError

    @staticmethod
    def Model(*args, **kwargs):
        require_mip_backend()

    @staticmethod
    def LinExpr(*args, **kwargs):
        require_mip_backend()

    @staticmethod
    def quicksum(*args, **kwargs):
        require_mip_backend()


HAS_MIP_BACKEND = _gurobi_module is not None
MIP_BACKEND_NAME = "gurobi" if HAS_MIP_BACKEND else None
grb = _gurobi_module if HAS_MIP_BACKEND else _UnavailableGRB()


def require_mip_backend(feature: str = "MIP") -> None:
    if HAS_MIP_BACKEND:
        return

    details = f" Details: {_BACKEND_IMPORT_ERROR}" if _BACKEND_IMPORT_ERROR else ""
    raise RuntimeError(
        f"{feature} requires an MILP backend, but no backend is available. "
        "Install Gurobi and make sure a valid license is available."
        f"{details}"
    )
