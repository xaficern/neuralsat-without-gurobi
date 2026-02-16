from __future__ import annotations

from .adapters import try_init_gurobi, try_init_highs

_SUPPORTED_BACKEND_NAMES = {"auto", "gurobi", "highs", "scipy"}


class BackendUnavailableError(RuntimeError):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def _normalize_backend_name(requested: str | None) -> str:
    requested = "auto" if requested is None else str(requested).strip().lower()
    if requested not in _SUPPORTED_BACKEND_NAMES:
        raise ValueError(
            f"Unsupported MILP backend '{requested}'. "
            "Expected one of: auto, gurobi, highs, scipy."
        )
    return requested


def _select_backend(requested: str):
    requested = _normalize_backend_name(requested)

    errors = []
    if requested in {"auto", "gurobi"}:
        ok, backend_module, err = try_init_gurobi()
        if ok:
            return True, "gurobi", backend_module, None
        errors.append(f"gurobi: {err}")
        if requested == "gurobi":
            return False, None, None, "; ".join(errors)

    if requested in {"auto", "highs", "scipy"}:
        ok, backend_module, err = try_init_highs()
        if ok:
            return True, "highs", backend_module, None
        errors.append(f"highs: {err}")
        if requested in {"highs", "scipy"}:
            return False, None, None, "; ".join(errors)

    return False, None, None, "; ".join(errors) if errors else None


class BackendRuntime:
    """Process-wide backend runtime state and access helpers."""

    def __init__(self):
        self._initialized = False
        self._requested = "auto"
        self._has_backend = False
        self._backend_name = None
        self._module = None
        self._error = None

    def _sync_legacy_globals(self):
        global HAS_MIP_BACKEND, MIP_BACKEND_NAME, _BACKEND_IMPORT_ERROR
        HAS_MIP_BACKEND = self._has_backend
        MIP_BACKEND_NAME = self._backend_name
        _BACKEND_IMPORT_ERROR = self._error

    def configure(self, requested: str = "auto") -> tuple[bool, str | None]:
        normalized = _normalize_backend_name(requested)
        has_backend, backend_name, backend_module, import_error = _select_backend(normalized)

        self._initialized = True
        self._requested = normalized
        self._has_backend = has_backend
        self._backend_name = backend_name
        self._module = backend_module
        self._error = import_error
        self._sync_legacy_globals()
        return has_backend, backend_name

    def ensure_initialized(self) -> None:
        if self._initialized:
            return
        self.configure("auto")

    @property
    def solver(self):
        self.ensure_initialized()
        if self._module is None:
            raise BackendUnavailableError(
                "MILP backend is unavailable. "
                "Configure a backend via Settings.mip_backend or --mip_backend."
            )
        return self._module

    def require(self, feature: str = "MIP") -> None:
        self.ensure_initialized()
        if self._has_backend:
            return

        details = f" Details: {self._error}" if self._error else ""
        raise RuntimeError(
            f"{feature} requires an MILP backend, but no backend is available. "
            "Install Gurobi with a valid license or install SciPy with HiGHS support."
            f"{details}"
        )

    def status(self) -> dict:
        self.ensure_initialized()
        return {
            "requested": self._requested,
            "available": self._has_backend,
            "active": self._backend_name,
            "error": self._error,
        }

    @property
    def has_backend(self) -> bool:
        self.ensure_initialized()
        return bool(self._has_backend)

    @property
    def backend_name(self) -> str | None:
        self.ensure_initialized()
        return self._backend_name


BACKEND = BackendRuntime()
HAS_MIP_BACKEND = False
MIP_BACKEND_NAME = None
_BACKEND_IMPORT_ERROR = None


# Compatibility wrappers.
def configure_backend(requested: str = "auto") -> tuple[bool, str | None]:
    return BACKEND.configure(requested)


def get_backend_status() -> dict:
    return BACKEND.status()


def has_mip_backend() -> bool:
    return BACKEND.has_backend


def get_mip_backend_name() -> str | None:
    return BACKEND.backend_name


def require_mip_backend(feature: str = "MIP") -> None:
    BACKEND.require(feature)
