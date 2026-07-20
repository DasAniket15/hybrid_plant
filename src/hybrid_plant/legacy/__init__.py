"""
Legacy optimisation path — superseded by the Pyomo LP in ``hybrid_plant.optimise``.

The Optuna TPE search in :mod:`hybrid_plant.legacy.solver_engine` was the original
optimiser. It is retained as a cross-check reference against the LP result and is
reachable only via ``solver.yaml → solver.engine: optuna``. The production default
is ``engine: pyomo``.

Nothing here is on the production path. It carries an optional dependency
(``optuna``), installed via the ``legacy`` or ``dev`` extras:

    pip install -e ".[legacy]"

New work belongs in ``hybrid_plant.optimise``, not here.
"""
