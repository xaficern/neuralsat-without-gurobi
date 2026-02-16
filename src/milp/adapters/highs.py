from __future__ import annotations

import copy
import math
from types import SimpleNamespace

import numpy as np


class BackendUnavailableError(RuntimeError):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

_HIGHS_RECOGNIZED_PARAMS = {
    "OutputFlag",
    "TimeLimit",
    "MIPGap",
    "BestBdStop",
    "BestObjStop",
    "Threads",
    "FeasibilityTol",
    "MIPGapAbs",
}
_HIGHS_WARN_ONLY_PARAMS = {"Threads", "FeasibilityTol", "MIPGapAbs"}


_GRB = SimpleNamespace(
    CONTINUOUS="CONTINUOUS",
    BINARY="BINARY",
    MINIMIZE=1,
    MAXIMIZE=-1,
    OPTIMAL=2,
    INFEASIBLE=3,
    UNBOUNDED=5,
    TIME_LIMIT=9,
    INTERRUPTED=11,
    USER_OBJ_LIMIT=15,
)


def _is_number(value) -> bool:
    return isinstance(value, (int, float, np.number))


class _LinExpr:
    def __init__(self, coeffs: dict[int, float] | None = None, const: float = 0.0):
        self.coeffs = dict(coeffs or {})
        self.const = float(const)

    def copy(self) -> "_LinExpr":
        return _LinExpr(coeffs=self.coeffs, const=self.const)

    def _add_inplace(self, other: "_LinExpr") -> "_LinExpr":
        self.const += other.const
        for var_idx, coeff in other.coeffs.items():
            self.coeffs[var_idx] = self.coeffs.get(var_idx, 0.0) + coeff
            if abs(self.coeffs[var_idx]) <= 1e-16:
                del self.coeffs[var_idx]
        return self

    def __add__(self, other):
        return self.copy()._add_inplace(_as_linexpr(other))

    def __radd__(self, other):
        return _as_linexpr(other).__add__(self)

    def __sub__(self, other):
        return self + (-_as_linexpr(other))

    def __rsub__(self, other):
        return _as_linexpr(other) - self

    def __mul__(self, other):
        if not _is_number(other):
            raise TypeError("Only scalar multiplication is supported in linear expressions.")
        other = float(other)
        return _LinExpr(
            coeffs={var_idx: coeff * other for var_idx, coeff in self.coeffs.items()},
            const=self.const * other,
        )

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * -1.0

    def __le__(self, other):
        return _TempConstr(self - other, "<=")

    def __ge__(self, other):
        return _TempConstr(self - other, ">=")

    def __eq__(self, other):  # type: ignore[override]
        return _TempConstr(self - other, "==")


class _TempConstr:
    def __init__(self, expr: _LinExpr, sense: str):
        self.expr = expr
        self.sense = sense


class _HiGHSVar:
    def __init__(self, model: "_HiGHSModel", idx: int, name: str, lb: float, ub: float, vtype: str):
        self._model = model
        self._idx = idx
        self._name = name
        self._lb = float(lb)
        self._ub = float(ub)
        self._vtype = vtype
        self._active = True
        self._x = None

    @property
    def VarName(self) -> str:
        return self._name

    @property
    def lb(self) -> float:
        return self._lb

    @lb.setter
    def lb(self, value: float):
        self._lb = float(value)

    @property
    def ub(self) -> float:
        return self._ub

    @ub.setter
    def ub(self, value: float):
        self._ub = float(value)

    @property
    def LB(self) -> float:
        return self._lb

    @LB.setter
    def LB(self, value: float):
        self._lb = float(value)

    @property
    def UB(self) -> float:
        return self._ub

    @UB.setter
    def UB(self, value: float):
        self._ub = float(value)

    @property
    def X(self) -> float:
        if self._x is None:
            return float("nan")
        return float(self._x)

    def _linexpr(self) -> _LinExpr:
        return _LinExpr(coeffs={self._idx: 1.0}, const=0.0)

    def __add__(self, other):
        return self._linexpr() + other

    def __radd__(self, other):
        return _as_linexpr(other) + self

    def __sub__(self, other):
        return self._linexpr() - other

    def __rsub__(self, other):
        return _as_linexpr(other) - self

    def __mul__(self, other):
        if not _is_number(other):
            raise TypeError("Only scalar multiplication is supported in linear expressions.")
        return self._linexpr() * float(other)

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return -self._linexpr()

    def __le__(self, other):
        return self._linexpr() <= other

    def __ge__(self, other):
        return self._linexpr() >= other

    def __eq__(self, other):  # type: ignore[override]
        return self._linexpr() == other


class _HiGHSConstr:
    def __init__(self, expr: _LinExpr, sense: str, name: str | None = None):
        self.expr = expr
        self.sense = sense
        self.ConstrName = name or ""
        self._active = True


def _as_linexpr(value) -> _LinExpr:
    if isinstance(value, _LinExpr):
        return value
    if isinstance(value, _HiGHSVar):
        return value._linexpr()
    if _is_number(value):
        return _LinExpr(const=float(value))
    raise TypeError(f"Unsupported expression type: {type(value)}")


class _HiGHSModel:
    def __init__(self, name: str | None = None):
        self._name = str(name) if name is not None else ""
        self._vars: list[_HiGHSVar] = []
        self._var_by_name: dict[str, _HiGHSVar] = {}
        self._constrs: list[_HiGHSConstr] = []
        self._objective = _LinExpr()
        self._objective_sense = _GRB.MINIMIZE
        self._params: dict[str, float | int | bool] = {}
        self._warned_params: set[str] = set()

        self.status = 0
        self.objbound = float("nan")
        self.objval = float("nan")
        self.solcount = 0

    @property
    def NumVars(self) -> int:
        return len([v for v in self._vars if v._active])

    @property
    def NumConstrs(self) -> int:
        return len([c for c in self._constrs if c._active])

    @property
    def ModelName(self) -> str:
        return self._name

    def copy(self):
        new_model = copy.deepcopy(self)
        for var in new_model._vars:
            var._model = new_model
        new_model._var_by_name = {var.VarName: var for var in new_model._vars if var._active}
        return new_model

    def update(self):
        return None

    def reset(self):
        self.status = 0
        self.objbound = float("nan")
        self.objval = float("nan")
        self.solcount = 0
        for var in self._vars:
            var._x = None
        return None

    def setObjective(self, expr_or_var, sense):
        self._objective = _as_linexpr(expr_or_var)
        self._objective_sense = sense

    def addVar(self, lb=-math.inf, ub=math.inf, obj=0.0, vtype=None, name: str | None = None):
        if vtype is None:
            vtype = _GRB.CONTINUOUS
        var_name = name or f"var_{len(self._vars)}"
        if var_name in self._var_by_name:
            raise BackendUnavailableError(f"Duplicate variable name: {var_name}")
        var = _HiGHSVar(self, idx=len(self._vars), name=var_name, lb=lb, ub=ub, vtype=vtype)
        self._vars.append(var)
        self._var_by_name[var_name] = var
        if abs(float(obj)) > 0:
            self._objective += float(obj) * var
        return var

    def addConstr(self, expr, name: str | None = None):
        if not isinstance(expr, _TempConstr):
            raise BackendUnavailableError("Constraint must be created via <=, >=, or == linear relation.")
        constr = _HiGHSConstr(expr=expr.expr, sense=expr.sense, name=name)
        self._constrs.append(constr)
        return constr

    def remove(self, objs):
        if objs is None:
            return
        if not isinstance(objs, (list, tuple, set)):
            objs = [objs]
        for obj in objs:
            if isinstance(obj, _HiGHSVar):
                obj._active = False
                self._var_by_name.pop(obj.VarName, None)
            elif isinstance(obj, _HiGHSConstr):
                obj._active = False
            else:
                raise BackendUnavailableError(f"Unsupported object passed to remove(): {type(obj)}")

    def getVarByName(self, name: str):
        var = self._var_by_name.get(name)
        if var is None or not var._active:
            return None
        return var

    def getVars(self):
        return [v for v in self._vars if v._active]

    def getConstrs(self):
        return [c for c in self._constrs if c._active]

    def _warn_unsupported_param(self, param_name: str):
        if param_name in self._warned_params:
            return
        self._warned_params.add(param_name)
        print(f"[!] HiGHS backend: parameter {param_name} is not directly supported and will be ignored.")

    def setParam(self, name: str, value):
        self._params[name] = value
        if name in _HIGHS_RECOGNIZED_PARAMS:
            if name in _HIGHS_WARN_ONLY_PARAMS:
                self._warn_unsupported_param(name)
            return
        self._warn_unsupported_param(name)

    def _active_var_index(self) -> tuple[list[_HiGHSVar], dict[int, int]]:
        active_vars = [var for var in self._vars if var._active]
        index = {var._idx: i for i, var in enumerate(active_vars)}
        return active_vars, index

    def _build_objective(self, active_vars, var_index):
        num_vars = len(active_vars)
        c = np.zeros(num_vars, dtype=np.float64)
        obj_const = self._objective.const
        for old_idx, coeff in self._objective.coeffs.items():
            if old_idx in var_index:
                c[var_index[old_idx]] += coeff

        sign = 1.0 if self._objective_sense == _GRB.MINIMIZE else -1.0
        c_solver = sign * c
        obj_const_solver = sign * obj_const
        return c_solver, sign, obj_const_solver

    def _build_bounds_and_integrality(self, active_vars):
        lb = np.array([var.lb for var in active_vars], dtype=np.float64)
        ub = np.array([var.ub for var in active_vars], dtype=np.float64)
        integrality = np.array(
            [1 if var._vtype == _GRB.BINARY else 0 for var in active_vars],
            dtype=np.int32,
        )
        return lb, ub, integrality

    def _build_constraints(self, num_vars, var_index):
        from scipy.optimize import LinearConstraint

        rows = []
        row_lb = []
        row_ub = []
        for constr in self._constrs:
            if not constr._active:
                continue
            row = np.zeros(num_vars, dtype=np.float64)
            for old_idx, coeff in constr.expr.coeffs.items():
                if old_idx in var_index:
                    row[var_index[old_idx]] += coeff
            rhs = -constr.expr.const
            if constr.sense == "<=":
                lb_i, ub_i = -np.inf, rhs
            elif constr.sense == ">=":
                lb_i, ub_i = rhs, np.inf
            elif constr.sense == "==":
                lb_i, ub_i = rhs, rhs
            else:
                raise BackendUnavailableError(f"Unsupported constraint sense: {constr.sense}")
            rows.append(row)
            row_lb.append(lb_i)
            row_ub.append(ub_i)

        constraints = []
        if rows:
            A = np.vstack(rows)
            constraints.append(
                LinearConstraint(
                    A=A,
                    lb=np.array(row_lb, dtype=np.float64),
                    ub=np.array(row_ub, dtype=np.float64),
                )
            )
        return constraints

    def _build_solver_options(self):
        options = {}
        if "TimeLimit" in self._params and self._params["TimeLimit"] is not None:
            options["time_limit"] = float(self._params["TimeLimit"])
        if "MIPGap" in self._params and self._params["MIPGap"] is not None:
            options["mip_rel_gap"] = float(self._params["MIPGap"])
        if "OutputFlag" in self._params:
            options["disp"] = bool(self._params["OutputFlag"])
        return options

    def _assign_solution_values(self, result, active_vars, var_index):
        self.solcount = 1 if getattr(result, "x", None) is not None else 0
        for var in self._vars:
            var._x = None
        if self.solcount > 0:
            for var in active_vars:
                var._x = float(result.x[var_index[var._idx]])

    def _set_status_from_result(self, result):
        if result.status == 0:
            self.status = _GRB.OPTIMAL
        elif result.status == 1:
            self.status = _GRB.TIME_LIMIT
        elif result.status == 2:
            self.status = _GRB.INFEASIBLE
        elif result.status == 3:
            self.status = _GRB.UNBOUNDED
        else:
            self.status = _GRB.INTERRUPTED

    def _update_objective_values(self, result, sign, obj_const_solver):
        if getattr(result, "fun", None) is not None:
            self.objval = sign * (float(result.fun) - obj_const_solver)
        else:
            self.objval = float("nan")

        if getattr(result, "mip_dual_bound", None) is not None:
            self.objbound = sign * (float(result.mip_dual_bound) - obj_const_solver)
        else:
            self.objbound = self.objval

    def _apply_user_stops(self):
        best_bd_stop = self._params.get("BestBdStop")
        if best_bd_stop is not None and np.isfinite(self.objbound):
            threshold = float(best_bd_stop)
            if self._objective_sense == _GRB.MINIMIZE and self.objbound >= threshold:
                self.status = _GRB.USER_OBJ_LIMIT
            if self._objective_sense == _GRB.MAXIMIZE and self.objbound <= threshold:
                self.status = _GRB.USER_OBJ_LIMIT

        best_obj_stop = self._params.get("BestObjStop")
        if best_obj_stop is not None and self.solcount > 0 and np.isfinite(self.objval):
            threshold = float(best_obj_stop)
            if self._objective_sense == _GRB.MINIMIZE and self.objval <= threshold:
                self.status = _GRB.USER_OBJ_LIMIT
            if self._objective_sense == _GRB.MAXIMIZE and self.objval >= threshold:
                self.status = _GRB.USER_OBJ_LIMIT

    def optimize(self):
        from scipy.optimize import Bounds, milp

        active_vars, var_index = self._active_var_index()
        num_vars = len(active_vars)
        if num_vars == 0:
            self.status = _GRB.OPTIMAL
            self.objbound = 0.0
            self.objval = 0.0
            self.solcount = 0
            return

        c_solver, sign, obj_const_solver = self._build_objective(active_vars, var_index)
        lb, ub, integrality = self._build_bounds_and_integrality(active_vars)
        constraints = self._build_constraints(num_vars, var_index)
        options = self._build_solver_options()

        result = milp(
            c=c_solver,
            constraints=constraints,
            integrality=integrality,
            bounds=Bounds(lb=lb, ub=ub),
            options=options,
        )

        self._assign_solution_values(result, active_vars, var_index)
        self._set_status_from_result(result)
        self._update_objective_values(result, sign, obj_const_solver)
        self._apply_user_stops()


def _highs_linexpr(coeffs, vars):
    expr = _LinExpr()
    coeff_arr = np.asarray(coeffs).reshape(-1)
    vars_list = list(vars)
    if len(coeff_arr) != len(vars_list):
        raise BackendUnavailableError("LinExpr coefficients and variables must have the same length.")
    for coeff, var in zip(coeff_arr, vars_list):
        if abs(float(coeff)) > 0:
            expr += float(coeff) * var
    return expr


def _highs_quicksum(terms):
    expr = _LinExpr()
    for term in terms:
        expr += term
    return expr


class HiGHSGrbShim:
    GRB = _GRB
    GurobiError = BackendUnavailableError
    Model = _HiGHSModel
    LinExpr = staticmethod(_highs_linexpr)
    quicksum = staticmethod(_highs_quicksum)


def try_init_highs():
    try:
        from scipy.optimize import milp

        return True, HiGHSGrbShim(), None
    except Exception as exc:
        return False, None, str(exc)
