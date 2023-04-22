import sys,os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import prepare
from scm_optimization.integer_dual_balancing import DualBalancing
from random import random,seed
from scm_optimization.model import *
from scipy.optimize import minimize, bisect, minimize_scalar
from dual_balancing_extension.simulation import Hospital_DB
import pandas as pd
from multiprocessing import Pool
import pickle
import numpy as np


time.time()


def run_sim(args):
    backlogging_cost = args['backlogging_cost']
    info = args['info']
    rep = args['rep']
    outdir = args['outdir']
    alpha = args['alpha']

    start_time = time.time()
    results = pd.DataFrame()

    seed(rep)
    np.random.seed(rep)

    lead_time = 0
    holding_cost = 1
    setup_cost = 0
    unit_price = 0
    gamma = 1

    #usage_model = PoissonUsageModel(scale=1, trunk=1e-3)
    usage_model = GenPoissonUsageModel(scale=1, alpha = 0,trunk=1e-3)
    info_state_rvs = [pacal.ConstDistr(0)] * info + \
                     [pacal.BinomialDistr(10, 0.5)]

    #if info == 0:
    #    info_state_rvs = [pacal.BinomialDistr(10, 0.5), pacal.ConstDistr(0)]

    model = DualBalancing(gamma,
                          lead_time,
                          info_state_rvs,
                          alpha,
                          holding_cost,
                          backlogging_cost,
                          setup_cost,
                          unit_price,
                          usage_model=usage_model)
                          

    print("backlogging cost:", backlogging_cost, " info: ", info, " rep: ", rep)

    hospital = Hospital_DB(db_model=model, periods=21)
    hospital.run()

    results = results.append({
        "info": info,
        "backlogging_cost": backlogging_cost,
        "rep": rep,
        "cost": hospital.cost_incurred,
        "backlog_cost_incurred": hospital.backlog_cost_incurred,
        "holding_cost_incurred": hospital.holding_cost_incurred,
        "schedule": hospital.schedule,
        "order_cont": hospital.order_continuous,
        "order": hospital.order,
        "demand": hospital.demand,
        "inventory": hospital.inventory_level,
        "run_time_min": (time.time() - start_time)/60
    }, ignore_index=True)
    
    return results


if __name__ == "__main__":
    import random
    outdir = sys.argv[1] if len(sys.argv) > 1 else "db_results"
    num_pools = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    backlogging_cost = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    info = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    reps = int(sys.argv[5])if len(sys.argv) > 5 else 10


    pool = Pool(num_pools)



    print(num_pools)

    alpha = 0.5
    backlogging_costs = [backlogging_cost]
    #backlogging_costs = [1000]
    infos = [info]
    #infos = [1]
    reps = list(range(reps))

    args_list = []
    for rep in reps:
        for backlogging_cost in backlogging_costs:
            for info in infos:
                args_list.append(
                    {"backlogging_cost": backlogging_cost,
                     "info": info,
                     "rep": rep,
                     "outdir":outdir,
                     'alpha':alpha}
                )

    results_files = []
    random.shuffle(args_list)

    results_files = pool.map(run_sim, args_list)
    #for arg in args_list:
    #    #run_sim(arg)
    #    results_files.append(run_sim(arg))

    all_results = pd.concat(results_files)
    all_results.to_csv("db_sim_{}_{}.csv".format(backlogging_cost,info), index=False)
