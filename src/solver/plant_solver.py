# src/solver/plant_solver.py

import numpy as np
import random


class PlantSolver:

    def __init__(self, config, data, energy_engine, finance_engine):

        self.config = config
        self.data = data
        self.energy_engine = energy_engine
        self.finance_engine = finance_engine

        self.solver_cfg = config.solver["solver"]
        self.debug = self.solver_cfg.get("debug_mode", False)

    # -------------------------------------------------

    def _sample_continuous(self, var_cfg):
        return random.uniform(var_cfg["min"], var_cfg["max"])

    def _sample_discrete(self, var_cfg):
        return random.randint(var_cfg["min"], var_cfg["max"])

    def _sample_choice(self, var_cfg):
        return random.choice(var_cfg["options"])

    # -------------------------------------------------

    def solve(self, n_samples=300, top_n=5):

        dv = self.solver_cfg["decision_variables"]
        target_irr = self.solver_cfg["constraints"]["minimum_roe_percent"]

        candidates = []

        for i in range(n_samples):

            # -----------------------------
            # 1️⃣ Sample decision variables
            # -----------------------------

            solar = self._sample_continuous(dv["solar_capacity_mw"])
            wind = self._sample_continuous(dv["wind_capacity_mw"])
            bess = self._sample_discrete(dv["bess_initial_containers"])

            charge_c = self._sample_continuous(dv["bess_charge_c_rate"])
            discharge_c = self._sample_continuous(dv["bess_discharge_c_rate"])

            ppa_cap = self._sample_continuous(dv["ppa_capacity_mw"])

            dispatch_priority = self._sample_choice(dv["dispatch_priority"])
            charge_source = self._sample_choice(dv["bess_charge_source"])
            banking_enabled = self._sample_choice(dv["banking_enabled"])

            # Augmentation placeholder (Phase-2)
            augmentation_trigger = self._sample_continuous(
                dv["bess_augmentation_trigger_percent"]
            )

            # -----------------------------
            # 2️⃣ Energy Engine
            # -----------------------------

            year1 = self.energy_engine.evaluate(
                solar_capacity_mw=solar,
                wind_capacity_mw=wind,
                bess_containers=bess,
                charge_c_rate=charge_c,
                discharge_c_rate=discharge_c,
                ppa_capacity_mw=ppa_cap,
                dispatch_priority=dispatch_priority,
                bess_charge_source=charge_source,
            )

            if year1.get("invalid_solution", False):
                continue

            # -----------------------------
            # 3️⃣ Finance Engine
            # -----------------------------

            finance = self.finance_engine.evaluate(
                year1_results=year1,
                solar_capacity_mw=solar,
                wind_capacity_mw=wind,
                target_irr_percent=target_irr,
            )

            if finance.get("invalid_solution", False):
                continue

            candidate = {
                "solar_mw": solar,
                "wind_mw": wind,
                "bess_containers": bess,
                "charge_c_rate": charge_c,
                "discharge_c_rate": discharge_c,
                "ppa_capacity_mw": ppa_cap,
                "dispatch_priority": dispatch_priority,
                "bess_charge_source": charge_source,
                "banking_enabled": banking_enabled,
                "required_ppa_tariff": finance["required_ppa_tariff"],
                "equity_irr": finance["achieved_equity_irr"],
                "savings_npv": finance["objective_value"],
            }

            candidates.append(candidate)

            if self.debug and len(candidates) % 20 == 0:
                print(f"Evaluated {len(candidates)} valid candidates")

        # -----------------------------
        # Rank candidates
        # -----------------------------

        ranked = sorted(
            candidates,
            key=lambda x: x["savings_npv"],
            reverse=True
        )

        top_candidates = ranked[:top_n]

        return {
            "top_candidates": top_candidates,
            "total_valid_evaluations": len(candidates),
        }