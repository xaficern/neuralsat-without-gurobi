def try_init_gurobi():
    try:
        import gurobipy as candidate_gurobi

        probe = candidate_gurobi.Model("neuralsat_backend_probe")
        probe.setParam("OutputFlag", 0)
        probe = None
        return True, candidate_gurobi, None
    except Exception as exc:
        return False, None, str(exc)
