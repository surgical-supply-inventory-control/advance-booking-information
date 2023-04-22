import sys, os, pacal, prepare
import textwrap
import os
import pickle
import pandas as pd
import pacal
import itertools
from datetime import date

policy_adress = ''
info_horizon = 2
data = pd.read_pickle(policy_adress)
leadtimes = set(data["lead_time"])
ts = [max(data["t"]) - i for i in range(7)]
bs = set(data["backlogging_cost"])


for l in leadtimes:
    for b in bs:
        ns_policy = [{} for _ in range(7)]
        for t in ts:
            rt = (-t - 1) % 7
            df = data[data["t"] == t]
            df = df[df["information_horizon"]==info_horizon]
            df = df[df["backlogging_cost"]==b]

            states = df["information_state"]
            order_up_level = df["order_up_to"]
            ns_policy[rt] = {state: (level, level-1) for state, level in zip(states, order_up_level)}

        fn = "scm_implementation/simulation_inputs/ns_policy_id_{}_b_{}_lt_{}_info_{}.pickle".format(item_id,
                                                                                            str(int(b)),
                                                                                            str(l),
                                                                                            str(info_horizon))
        with open(fn, "wb") as f:
            pickle.dump(ns_policy, f)
            print(fn)
            print(ns_policy)



