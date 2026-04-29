"""Entry point: `python -m hybrid_plant` delegates to `python -m hybrid_plant.run_model`."""
import runpy
runpy.run_module("hybrid_plant.run_model", run_name="__main__", alter_sys=True)
