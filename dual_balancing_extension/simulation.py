import sys,os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import prepare
from scm_optimization.integer_dual_balancing import DualBalancing
from random import random
from scm_optimization.model import *
from scipy.optimize import minimize, bisect, minimize_scalar
import pandas as pd
import pickle
import glob

class Hospital_DB:
    def __init__(self, db_model, periods=20):
        self.db_model = db_model
        self.n_info = len(db_model.info_state_rvs) - 1
        self.periods = periods

        self.schedule = [0] + list(sum(self.db_model.info_state_rvs).rand(n=periods))
        self.demand = [db_model.usage_model.random(x,db_model.alpha) for x in self.schedule]

        self.order = [0] * (periods + 1)
        self.order_continuous = [0] * (periods + 1)
        self.inventory_level = [0] * (periods + 1)
        # self.inventory_position = [0] * (periods + 1)
        self.cost_incurred = 0
        self.backlog_cost_incurred = 0
        self.holding_cost_incurred = 0

        self.clock = 1

    def clock_to_time(self, clock):
        time = self.periods - clock
        return time

    def run(self):
        x = self.inventory_level[self.clock - 1]
        o = tuple(self.schedule[self.clock: self.clock + self.n_info])
        i_max = len(self.schedule)
        while self.clock < len(self.schedule):
            t = self.clock_to_time(self.clock)
            q = self.db_model.order_q_continuous(t, x, o)

            self.order_continuous[self.clock] = q
            order_q = int(q) if random() > q - int(q) else int(q) + 1
            self.order[self.clock] = order_q

            x += order_q - self.demand[self.clock]
            self.inventory_level[self.clock] = x

            self.cost_incurred += self.db_model.h * max([0, x]) - self.db_model.b * min([0, x])
            self.backlog_cost_incurred -= self.db_model.b * min([0, x])
            self.holding_cost_incurred += self.db_model.h * max([0, x])
            self.clock += 1

            o = tuple(self.schedule[self.clock: min([self.clock + self.n_info, i_max])])


class Hospital_LA:
    def __init__(self, model, periods=20):
        self.model = model
        self.n_info = len(model.info_state_rvs) - 1
        self.periods = periods

        self.schedule = [0] + list(sum(self.model.info_state_rvs).rand(n=periods))
        self.demand = [model.usage_model.random(x) for x in self.schedule]

        self.order = [0] * (periods + 1)
        self.order_continuous = [0] * (periods + 1)
        self.inventory_level = [0] * (periods + 1)
        # self.inventory_position = [0] * (periods + 1)
        self.cost_incurred = 0
        self.backlog_cost_incurred = 0
        self.holding_cost_incurred = 0

        self.clock = 1

    def clock_to_time(self, clock):
        time = self.periods - clock
        return time

    def run(self):
        x = self.inventory_level[self.clock - 1]
        o = tuple(self.schedule[self.clock: self.clock + self.n_info])
        i_max = len(self.schedule)
        while self.clock < len(self.schedule):
            t = self.clock_to_time(self.clock)
            order_q = self.model.order_la (t, x, o)
            self.order[self.clock] = order_q

            x += order_q - self.demand[self.clock]
            self.inventory_level[self.clock] = x

            self.cost_incurred += self.model.h * max([0, x]) - self.model.b * min([0, x])
            self.backlog_cost_incurred -= self.model.b * min([0, x])
            self.holding_cost_incurred += self.model.h * max([0, x])
            self.clock += 1
            o = tuple(self.schedule[self.clock: min([self.clock + self.n_info, i_max])])
