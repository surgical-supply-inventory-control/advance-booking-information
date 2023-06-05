import sys, os, pacal 
from pathlib import Path

directory = r'C:\Users\gorgu\Dropbox\Surgical_Inventory\Github Organize'
sys.path.insert(0, directory)
os.chdir(directory)

from scm_simulation.hospital import Hospital, EmpiricalElectiveSurgeryDemandProcess, \
    EmpiricalEmergencySurgeryDemandProcess, ParametricEmergencySurgeryDemandProcessWithPoissonUsage, \
    ParametricElectiveSurgeryDemandProcessWithPoissonUsage, \
    ParametricEmergencySurgeryDemandProcessWithTruncatedPoissonUsage, \
    ParametricElectiveSurgeryDemandProcessWithTruncatedPoissonUsage, \
    ParametricEmergencySurgeryDemandProcessWithTruncatedGeneralizedPoissonUsage, \
    ParametricElectiveSurgeryDemandProcessWithTruncatedGeneralizedPoissonUsage
from scm_optimization.integer_dual_balancing import DualBalancing
from scm_optimization.non_stationary_integer_dual_balancing import NonStationaryDualBalancing
from scm_optimization.model import *
from scm_simulation.item import Item
from scm_simulation.surgery import Surgery
from scm_simulation.rng_classes import GeneratePoisson, GenerateFromSample, GenerateDeterministic
from scm_simulation.order_policy import AdvancedInfoSsPolicy,  LAPolicy
import pickle
import pandas as pd
import numpy as np
from multiprocessing import Pool
from datetime import datetime, date
from scm_implementation.ns_info_state_rvs.ns_info_state_rvs import elective_info_rvs, emergency_info_rvs,alphas
from itertools import repeat
import argparse

"""
Elective and emergency surgeries per day are empirically generated
Surgery definition are empirically generated (random sampling from all historical surgeries)
Surgery item usage are based on historical
"""


def run(args):
    seed = 0
    item_id, b, n, lt, seed = args

    fn = "scm_implementation/simulation_inputs/ns_policy_id_{}_b_{}_lt_{}_info_{}.pickle".format(item_id, b, lt, n)
    with open(fn, "rb") as f:
        policy = pickle.load(f)

    policy = {item_id: AdvancedInfoSsPolicy(item_id, policy)}
    order_lt = {item_id: GenerateDeterministic(lt)}

    elective_process = ParametricElectiveSurgeryDemandProcessWithTruncatedGeneralizedPoissonUsage(seed=seed)
    emergency_process = ParametricEmergencySurgeryDemandProcessWithTruncatedGeneralizedPoissonUsage(seed=seed)

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


def run_la_policy(args):
    start_time = time.time()
    item_id, b, n, lt, seed = args
    
    elective_info_state_rv = elective_info_rvs[item_id]
    emergency_info_state_rv = emergency_info_rvs[item_id]
    
    horizon = n
    if elective_info_state_rv and emergency_info_state_rv:
        ns_info_state_rvs = []
        # 0..4 Weekdays, 5, 6 weekdays
        for rt in range(7):
            t_rt = (rt + horizon) % 7
            info_state_rv = [emergency_info_state_rv] + [pacal.ConstDistr(0)] * horizon
            if t_rt not in [5, 6]:
                info_state_rv[-1] += elective_info_state_rv
            if horizon == 0:
                info_state_rv += [pacal.ConstDistr(0)]
            ns_info_state_rvs.append(info_state_rv)

    usage_model = GenPoissonUsageModel(scale=1, trunk=1e-3)
    la_model = NonStationaryDualBalancing(1,
                             lt,
                             ns_info_state_rvs,
                             alphas[item_id],
                             1,
                             b,
                             0,
                             0,
                             usage_model=usage_model)

    policy = {item_id: LAPolicy(item_id, la_model=la_model)}
    order_lt = {item_id: GenerateDeterministic(lt)}

    elective_process = ParametricElectiveSurgeryDemandProcessWithTruncatedGeneralizedPoissonUsage(seed=seed)
    emergency_process = ParametricEmergencySurgeryDemandProcessWithTruncatedGeneralizedPoissonUsage(seed=seed)

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
    r = {"item_id": [item_id],
         "backlogging_cost": [b],
         "info_horizon": [n],
         "lead_time": [lt],
         "average_inventory_level": [np.mean(hospital.full_inventory_lvl[item_id])],
         "full_inventory_lvl": [hospital.full_inventory_lvl[item_id]],
         "surgeries_backlogged": [stock_outs],
         "service_level": [service_level],
         "seed": [seed],
         "run_time_min": [(time.time() - start_time) / 60]
         }
    print("Finished: ", datetime.now().isoformat(), "-", item_id, b, n, seed)

    results_fn = "la_case_study_results_item_{}_lt_{}_b_{}_seed_{}_{}.csv".format(
                                                                   item_id,
                                                                    str(lt),
                                                                   str(b),
                                                                   str(seed),
                                                                   datetime.now().strftime("%Y-%m-%d_%H%M_%S_%f")
                                                                   )

    pd.DataFrame(r).to_csv(results_fn, index=False)
    return r

class keyvalue(argparse.Action):
    # Constructor calling
    def __call__( self , parser, namespace,
                 values, option_string = None):
        setattr(namespace, self.dest, dict())
          
        for value in values:
            # split it into key and value
            key, value = value.split('=')
            # assign into dictionary
            getattr(namespace, self.dest)[key] = value

    def halfwidth(series):
        return 1.96 * np.std(series) / np.sqrt(len(series))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
      
    # adding an arguments 
    parser.add_argument('--kwargs', 
                        nargs='*', 
                        action = keyvalue)
      
     #parsing arguments 
    args = parser.parse_args().kwargs
    item = args['item']
    b = int(args['b'])
    lt = int(args['lt'])
    pool_ = int(args['pool'])

    pool = Pool(pool_)
    results = pd.DataFrame()

    item_ids = [item]
    bs = [b]
    lts = [lt]
    ns = [0]

    all_args = []

    for item_id in item_ids:
        for lt in lts:
            for b in bs:
                for n in ns:
                    for seed in range(0, 16):
                        all_args.append((item_id, b, n, lt, seed))
    
    rs = pool.map(run_la_policy, all_args)
    for r in rs:
        results = results.append(r, ignore_index=True)

    results.to_pickle(str(date.today()) + "_parametric_case_study_results.pickle")
    results.to_csv(str(date.today()) + "_parametric_case_study_results.csv")

   
