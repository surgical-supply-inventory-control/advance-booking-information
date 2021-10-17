from scm_simulation.hospital import Hospital,ParametricEmergencySurgeryDemandProcessWithTruncatedPoissonUsage, \
    ParametricElectiveSurgeryDemandProcessWithTruncatedPoissonUsage
from scm_optimization.model import *
from scm_simulation.rng_classes import GenerateDeterministic
from scm_simulation.order_policy import AdvancedInfoSsPolicy
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

"""
Elective and emergency surgeries per day are empirically generated
Surgery definition are empirically generated (random sampling from all historical surgeries)
Surgery item usage are based on historical
"""


def run(args):
    seed = 0
    item_id, b, n, lt, seed = args

    fn = "model_artifact/case_study_policy/ns_policy_id_{}_b_{}_lt_{}_info_{}.pickle".format(item_id, b, lt, n)
    with open(fn, "rb") as f:
        policy = pickle.load(f)

    policy = {item_id: AdvancedInfoSsPolicy(item_id, policy)}
    order_lt = {item_id: GenerateDeterministic(lt)}
    elective_process = ParametricElectiveSurgeryDemandProcessWithTruncatedPoissonUsage(seed=seed)
    emergency_process = ParametricEmergencySurgeryDemandProcessWithTruncatedPoissonUsage(seed=seed)

    hospital = Hospital([item_id],
                        policy,
                        order_lt,
                        emergency_process,
                        elective_process,
                        warm_up=7,
                        sim_time=365,
                        end_buffer=7)

    hospital.run_simulation()
    hospital.trim_data()

    stock_outs = sum(len(d) for d in hospital.full_surgery_backlog)
    service_level = sum(len(d) for d in hospital.full_elective_schedule) \
                    + sum(len(d) for d in hospital.full_emergency_schedule)
    service_level = 1 - stock_outs / service_level
    r = {"item_id": item_id,
         "backlogging_cost": b,
         "info_horizon": n,
         "lead_time": lt,
         "average_inventory_level": np.mean(hospital.full_inventory_lvl[item_id]),
         "full_inventory_lvl": hospital.full_inventory_lvl[item_id],
         "surgeries_backlogged": stock_outs,
         "service_level": service_level,
         "seed": seed
         }
    print("Finished: ", datetime.now().isoformat(), "-", item_id, b, n, seed)
    return r


if __name__ == "__main__":
    """
        item id: 1686 | 47320 | 38197 | 21920 | 82099
        b: 100 | 1000 | 10000
        lt: 0 | 1
        n: 0 | 1 | 2
        seed: int
     """
    item_id = "1686"
    b = 100
    lt = 1
    n = 2
    seed = 0
    results = pd.DataFrame()
    arg = (item_id, b, n, lt, seed)
    result = run(arg)
    pprint(result)
