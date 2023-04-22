from scm_optimization.integer_dual_balancing import DualBalancing
from scm_optimization.model import *
from scipy.optimize import minimize, bisect, minimize_scalar
from functools import lru_cache


class LA_DB_Model:
    @classmethod
    def read_pickle(cls, filename):
        with open(filename, "rb") as f:
            m = pickle.load(f)
        return m

    def __init__(self,
                 gamma,
                 lead_time,
                 ns_info_state_rvs,
                 alpha,
                 holding_cost,
                 backlogging_cost,
                 setup_cost,
                 unit_price,
                 usage_model=None,
                 increments=1,
                 detailed=False
                 ):



        # parameters in order:
        # single period discount factor
        # lead time for items to arrive, >= 0
        # information horizon N >= 0, N = 0 for no advanced information
        # vector of random variables, transition of the state of advanced information, M_{t, s} in notation
        self.detailed = detailed
        self.gamma = gamma
        self.lead_time = lead_time
        self.ns_info_state_rvs = ns_info_state_rvs
        self.period = len(ns_info_state_rvs)
        self.info_horizon = max(self.lead_time, len(ns_info_state_rvs[0]))
        self.increments = increments

        # usage_model = lambda o: pacal.BinomialDistr(o, p=0.5)
        # usage_model = lambda o: pacal.ConstDistr(o)
        # usage_model = usage_model=pacal.PoissonDistr
        default_usage_model = PoissonUsageModel(scale=1)
        self.usage_model = usage_model if usage_model else default_usage_model
        self.ns_unknown_lt_demand_rvs = []
        self.ns_unknown_demand_rvs = []
        self.alpha = alpha

        self.h = holding_cost
        self.b = backlogging_cost
        self.k = setup_cost
        self.c = unit_price



        # static list of possible info states
        self.info_states_cache = None
        self.info_states_prob_cache = {}

        # all caches
        self.value_function_j = {}
        self.j_h = {}
        self.j_b = {}
        self.j_k = {}
        self.j_p = {}

        self.value_function_v = {}
        self.v_h = {}
        self.v_b = {}
        self.v_k = {}
        self.v_p = {}

        self.value_function_v_argmin = {}
        self.base_stock_level_cache = {}
        self.current_demand_cache = {}

        self.reward_funcion_g_cache = {}
        self.g_h = {}
        self.g_b = {}
        self.g_p = {}

        self.order_la_cache = {}
        self.base_stock_la_cache = {}

        if len(self.ns_info_state_rvs[0]) < self.lead_time + 1:
            diff = self.lead_time - len(self.ns_info_state_rvs[0]) + 1
            self.ns_info_state_rvs = list(info_state_rvs + diff * [pacal.ConstDistr(0)]
                                          for info_state_rvs in self.ns_info_state_rvs)

        for k in range(self.period):
            unknown_lt_info_rv = sum(self.ns_info_state_rvs[(j + k) % self.period][i]
                                     for j in range(self.lead_time + 1) for i in range(j + 1))
            if len(unknown_lt_info_rv.get_piecewise_pdf().getDiracs()) == 1:
                v = unknown_lt_info_rv.get_piecewise_pdf().getDiracs()[0].a
                self.ns_unknown_lt_demand_rvs.append(self.usage_model.usage(v,self.alpha) if v else pacal.ConstDistr(0))
            else:
                unknown_lt_demand_pdf = sum([dirac.f * self.usage_model.usage(dirac.a,self.alpha).get_piecewise_pdf()
                                             for dirac in unknown_lt_info_rv.get_piecewise_pdf().getDiracs()])
                unknown_lt_demand_rv = pacal.DiscreteDistr([dirac.a for dirac in unknown_lt_demand_pdf.getDiracs()],
                                                           [dirac.f for dirac in unknown_lt_demand_pdf.getDiracs()])
                self.ns_unknown_lt_demand_rvs.append(unknown_lt_demand_rv)
                                          
        ### Apppend Const(0) to info_state_rvs if leadtime > info_horizon


        for k in range(self.period):
            unknown_info_rv = self.ns_info_state_rvs[k][0]
            if len(unknown_info_rv.get_piecewise_pdf().getDiracs()) == 1:
                val = unknown_info_rv.get_piecewise_pdf().getDiracs()[0].a
                self.ns_unknown_demand_rvs.append(self.usage_model.usage(val,self.alpha) if val else pacal.ConstDistr(0))
            else:
                unknown_demand_pdf = sum([dirac.f * self.usage_model.usage(dirac.a,self.alpha).get_piecewise_pdf()
                                          for dirac in unknown_info_rv.get_piecewise_pdf().getDiracs()])
                unknown_demand_rv = pacal.DiscreteDistr([dirac.a for dirac in unknown_demand_pdf.getDiracs()],
                                                        [dirac.f for dirac in unknown_demand_pdf.getDiracs()])
                self.ns_unknown_demand_rvs.append(unknown_demand_rv)

        self.ns_no_info_demand = []
        self.ns_unknown_demand_cache = []
        for k in range(self.period):
            no_info_booking = sum(self.ns_info_state_rvs[k])
            no_info_demand_pdf = sum([dirac.f * self.usage_model.usage(dirac.a,self.alpha).get_piecewise_pdf()
                                    for dirac in no_info_booking.get_piecewise_pdf().getDiracs()
                                    ])
            self.no_info_demand = pacal.DiscreteDistr([dirac.a for dirac in no_info_demand_pdf.getDiracs()],
                                                    [dirac.f for dirac in no_info_demand_pdf.getDiracs()])
            self.ns_no_info_demand.append(self.no_info_demand)

            self.unknown_demand_cache = []
            prev = 0
            for i in range(0, len(self.ns_info_state_rvs[k])):
                booking_rv = sum(self.ns_info_state_rvs[k][0:i + 1])
                demand_pdf = sum([dirac.f * self.usage_model.usage(dirac.a,self.alpha).get_piecewise_pdf()
                                for dirac in booking_rv.get_piecewise_pdf().getDiracs()
                                ])
                demand_rv = pacal.DiscreteDistr([dirac.a for dirac in demand_pdf.getDiracs()],
                                                [dirac.f for dirac in demand_pdf.getDiracs()])
                demand_rv += prev
                #demand_rv = self.trunk_rv(demand_rv)
                self.unknown_demand_cache.append(demand_rv)
                prev = demand_rv
            self.ns_unknown_demand_cache.append(self.unknown_demand_cache)
        self.demand_rv_cache = {}  # (periods, o) -> RV
        self.q_cache = {}


    def rt(self, t):
        return (- t - 1) % self.period

    def trunk_rv(self, rv, trunk=1e-5):
        a = [dirac.a for dirac in rv.get_piecewise_pdf().getDiracs()]
        f = [dirac.f for dirac in rv.get_piecewise_pdf().getDiracs()]
        head = 0
        tail = 0
        while head < trunk / 2 and f[0] < trunk / 2:
            head += f[0]
            a.pop(0)
            f.pop(0)
        while tail < trunk and f[-1] < trunk:
            tail += f[-1]
            a.pop(-1)
            f.pop(-1)
        rv = pacal.DiscreteDistr([aa for aa in a],
                                 [ff/(1-head-tail) for ff in f])
        return rv

    def window_demand(self, t, j, o):
        print("in window_demand", t, j, o)
        """
        :param t: current period, dummy variable for stationary case
        :param j: end period (inclusive) 0 is the last period.
        :param o: info state
        :return: demand rv, cumulative demand RV for periods t to j given information o
        """
        periods = t - j + 1
        cumul_o = sum(o[0:min(len(o), periods)])
        if (periods, cumul_o) in self.demand_rv_cache:
            return self.demand_rv_cache[(periods, cumul_o)]

        self.unknown_demand_cache = self.ns_unknown_demand_cache[self.cur_period]
        self.no_info_demand = self.ns_no_info_demand[self.cur_period]

        while len(self.unknown_demand_cache) < periods:
            rv = self.unknown_demand_cache[-1] + self.no_info_demand
            rv = pacal.DiscreteDistr([dirac.a for dirac in rv.get_piecewise_pdf().getDiracs()],
                                     [dirac.f for dirac in rv.get_piecewise_pdf().getDiracs()])
            #rv = self.trunk_rv(rv)
            self.unknown_demand_cache.append(rv)
        index = periods - 1
        rv = self.usage_model.usage(cumul_o,self.alpha) + self.unknown_demand_cache[index]
        rv = self.trunk_rv(rv)
        #rv = pacal.DiscreteDistr([dirac.a for dirac in rv.get_piecewise_pdf().getDiracs()],
        #                         [dirac.f for dirac in rv.get_piecewise_pdf().getDiracs()])

        self.demand_rv_cache[(periods, cumul_o)] = rv
        print(rv.get_piecewise_pdf().getDiracs())
        return self.demand_rv_cache[(periods, cumul_o)]

    def h_db(self, q, t, x, o):
        print(" in h_db", q, t, x, o)
        """"Expected holding cost incurred by order size of q by those units from t to end of horizon
            state space is t for current priod
            o for info state
            x for current inventory position
        """
        expected_cost = 0
        s = t - self.lead_time
        while s >= 0:
            over_stock = pacal.max(0,
                                   q - pacal.max(0,
                                                 self.window_demand(t, s, o) - pacal.ConstDistr(x)
                                                 )
                                   )
            diff = over_stock.mean() * self.h
            if diff < 0.01:
                #print(t, s)
                break
            expected_cost += over_stock.mean() * self.h
            s -= 1

        return expected_cost

    def pi_db(self, q, t, x, o):
        s = t - self.lead_time
        under_stock = pacal.max(0, self.window_demand(t, s, o) - x - q)
        expected_cost = under_stock.mean() * self.b
        return expected_cost

    def la_objective(self, q, t, x, o):
        return self.h_db(q, t, x, o) + self.pi_db(q, t, x, o)

    # def order_la(self, t, x, o,clock):
    #     print(t, x, o)
    #     self.cur_period = clock%7
    #     prev, cost = float('inf'), float('inf')
    #     for q in range(0, 50):
    #         cost = self.la_objective(q, t, x, o)
    #         if cost < prev:
    #             prev = cost
    #         else:
    #             print("order_la:")
    #             print("    args (t, x, o):", t, x, o)
    #             print("    q:", q-1)
    #             return q-1
    #     print("MAXIMUM HIT: ERROR")


    def order_la(self, t, x, o):
        self.cur_period = self.rt(t)
        if (t, x, o) in self.order_la_cache:
            return self.order_la_cache[(t, x, o)]
        if (t, o) in self.base_stock_la_cache:
            q = max([self.base_stock_la_cache[(t, o)] - x, 0])
            return q

        prev, cost = float('inf'), float('inf')
        for q in range(0, 50):
            cost = self.la_objective(q, t, x, o)
            if cost < prev:
                prev = cost
            else:
                q = q - 1
                self.order_la_cache[(t, x, o)] = q
                if x == 0:
                    self.base_stock_la_cache[(t, o)] = q
                return q
        print("MAXIMUM HIT: ERROR")

        return q

    # @lru_cache()
    def g_objective(self, q, t, x, o):
        #print("in g_objective")
        return max([self.h_db(q, t, x, o),
                    self.pi_db(q, t, x, o)])

    # @lru_cache()
    def order_q_continuous(self, t, x, o):
        print(t, x, o)
        if (t, x, 0) in self.q_cache:
            return self.q_cache[(x, t, o)]

        q = minimize_scalar(lambda qq: self.g_objective(qq, t, x, o),
                               bracket=[0, 50],
                               bounds=[0, 50],
                               method='Bounded',
                               options={'xatol': 1e-03, 'maxiter': 500, 'disp': 0}).x
        self.q_cache[(x, t, o)] = q
        return q

    # @staticmethod
    # def abs_rv(rv):
    #     rv.get()


    def info_states(self):
        if self.info_states_cache:
            return self.info_states_cache
        info_horizon = len(self.ns_info_state_rvs[0])
        info_states_list = []
        info_states_list_p = []
        for k in range(self.period):
            info_states = []
            info_states_p = []
            for o in range(info_horizon - 1):
                relevant_rvs = [self.ns_info_state_rvs[(k - i) % self.period][i + o]
                                for i in range(1, info_horizon-o)]
                info_vals = [[diracs.a for diracs in rv.get_piecewise_pdf().getDiracs()] for rv in relevant_rvs]
                info_vals_p = [[diracs.f for diracs in rv.get_piecewise_pdf().getDiracs()] for rv in relevant_rvs]

                all_info_states = list(sum(c) for c in itertools.product(*info_vals))
                all_info_states_p = list(numpy.product(c) for c in itertools.product(*info_vals_p))

                new_info_states_p = {}
                for v, p in zip(all_info_states, all_info_states_p):
                    if v in new_info_states_p:
                        new_info_states_p[v] = new_info_states_p[v] + p
                    else:
                        new_info_states_p[v] = p

                #info_states.append(set(sum(c) for c in itertools.product(*info_vals)))
                info_states.append(list(new_info_states_p.keys()))
                info_states_p.append(list(new_info_states_p.values()))

            info_states_list.append(list(itertools.product(*info_states)))
            info_states_list_p.append(
                {v: numpy.product(ps) for v, ps in
                 zip(itertools.product(*info_states), itertools.product(*info_states_p))}
            )
        self.info_states_cache = info_states_list
        self.info_states_prob_cache = info_states_list_p
        return self.info_states_cache

    def get_info_state_prob(self, t, o):
        rt = self.rt(t)
        if self.info_states_prob_cache:
            return self.info_states_prob_cache[rt][o]
        else:
            self.info_states()
            return self.info_states_prob_cache[o]
    # def lambda_t(self, o):
    #     return o[0] + self.info_state_rvs[-1]
    # # O_t^L in notation
    # def observed_lt_info(self, o):
    #     return sum(o[0: self.lead_time + 1])
    #
    # # U_t^L in notation
    # def unobserved_lt_info(self, o):
    #     return sum(self.info_state_rvs[self.lead_time:])
    #
    # # \Lambda_t^L in notation
    # def lt_info_state(self, o):
    #     return self.observed_lt_info(o) + self.unobserved_lt_info(o)

    # D_t^L | \Lambda_t in notation


    def lt_demand(self, rt, lt_o):
        lt_demand_rv = self.usage_model.usage(lt_o,self.alpha) if lt_o else pacal.ConstDistr(0)
        if self.ns_unknown_lt_demand_rvs[self.cur_period][rt]:
            lt_demand_rv += self.ns_unknown_lt_demand_rvs[self.cur_period][rt]
        return lt_demand_rv

    def current_demand(self, rt, o):
        if (rt, o[0]) in self.current_demand_cache:
            return self.current_demand_cache[(rt, o[0])]
        if self.ns_unknown_demand_rvs[self.cur_period][rt]:
            current_demand = self.usage_model.usage(o[0],self.alpha) + self.ns_unknown_demand_rvs[self.cur_period][rt]
        else:
            current_demand = self.usage_model.usage(o[0],self.alpha)
        self.current_demand_cache[(rt, o[0])] = current_demand
        return current_demand

    # D_t | \Lambda_t in notation

    # expected discounted holding and backlog cost at end of period t + L given action (target inventory position)
    # # \tilde{G} in notation
    # def G_future(self, y, lt_o):
    #     x = y - self.lt_demand(lt_o)
    #     h_cost = self.h * pacal.max(x, 0).mean()
    #     b_cost = -self.b * pacal.min(x, 0).mean()
    #     cost = h_cost + b_cost
    #     v = self.gamma ** self.lead_time * cost

    #     self.g_b[(y, lt_o)] = self.gamma ** self.lead_time * b_cost
    #     self.g_h[(y, lt_o)] = self.gamma ** self.lead_time * h_cost
    #     return v

    # def G_b(self, y, o):
    #     self.G(y, o)
    #     lt_o = sum(o[0:self.lead_time + 1])
    #     return self.g_b[(y, lt_o)]

    # def G_h(self, y, o):
    #     self.G(y, o)
    #     lt_o = sum(o[0:self.lead_time + 1])
    #     return self.g_h[(y, lt_o)]

    # def G_p(self, y, o):
    #     self.G(y, o)
    #     lt_o = sum(o[0:self.lead_time + 1])
    #     return self.g_p[(y, lt_o)]

    # def G(self, y, o):
    #     lt_o = sum(o[0:self.lead_time + 1])
    #     if (y, lt_o) in self.reward_funcion_g_cache:
    #         return self.reward_funcion_g_cache[(y, lt_o)]

    #     self.reward_funcion_g_cache[(y, lt_o)] = (1 - self.gamma) * self.c * y + self.G_future(y, lt_o)
    #     self.g_p[(y, lt_o)] = (1 - self.gamma) * self.c * y
    #     return self.reward_funcion_g_cache[(y, lt_o)]


    def G_future(self, rt, y, lt_o):
            x = y - self.lt_demand(rt, lt_o)
            h_cost = self.h * pacal.max(x, 0).mean()
            b_cost = -self.b * pacal.min(x, 0).mean()
            cost = h_cost + b_cost
            v = self.gamma ** self.lead_time * cost

            self.g_b[(rt, y, lt_o)] = self.gamma ** self.lead_time * b_cost
            self.g_h[(rt, y, lt_o)] = self.gamma ** self.lead_time * h_cost
            return v

    def G_b(self, rt, y, o):
        self.G(rt, y, o)
        lt_o = sum(o[0:self.lead_time + 1])
        return self.g_b[(rt, y, lt_o)]

    def G_h(self, rt, y, o):
        self.G(rt, y, o)
        lt_o = sum(o[0:self.lead_time + 1])
        return self.g_h[(rt, y, lt_o)]

    def G_p(self, rt, y, o):
        self.G(rt, y, o)
        lt_o = sum(o[0:self.lead_time + 1])
        return self.g_p[(rt, y, lt_o)]

    def G(self, rt, y, o):
        lt_o = sum(o[0:self.lead_time + 1])
        if (rt, y, lt_o) in self.reward_funcion_g_cache:
            return self.reward_funcion_g_cache[(rt, y, lt_o)]

        self.reward_funcion_g_cache[(rt, y, lt_o)] = (1 - self.gamma) * self.c * y + self.G_future(rt, y, lt_o)
        self.g_p[(rt, y, lt_o)] = (1 - self.gamma) * self.c * y
        return self.reward_funcion_g_cache[(rt, y, lt_o)]

    def state_transition(self, t, y, o):
        rt = self.rt(t)
        next_x = y - self.current_demand(rt, o)
        next_o = [i + j for i, j in zip(self.ns_info_state_rvs[rt][1:], o[1:] + (0,))]
        return t - 1, next_x, next_o

    # def unpack_state_transition(self, t, x_rv, o_rv):
    #     states = []
    #     probabilities = []

    #     x_diracs = x_rv.get_piecewise_pdf().getDiracs()
    #     o_diracs = [o.get_piecewise_pdf().getDiracs() for o in o_rv]

    #     o_combinations = list(itertools.product(*o_diracs))
    #     for next_x in x_diracs:
    #         for next_o in o_combinations:
    #             p = next_x.f * numpy.prod([info.f for info in next_o])
    #             states.append((t, next_x.a, tuple(info.a for info in next_o)))
    #             probabilities.append(p)
    #     return states, probabilities


    # def j_function_la(self, t, x, o):
    #     if (t, x, o) in self.value_function_j:
    #         return self.value_function_j[(t, x, o)]
    #     elif t == -1:
    #         return 0
    #     else:
    #         # Exploit S s structure
    #         stock_up_lvl, base_stock_lvl = self.stock_up_level(t, o), self.base_stock_level(t, o)
    #         y = stock_up_lvl if x <= base_stock_lvl else x
    #         k = self.k if x <= base_stock_lvl else 0
    #         j_value = k + self.v_function(t, y, o)

    #         self.value_function_j[(t, x, o)] = j_value
    #         if self.detailed:
    #             j_b = self.v_b[(t, y, o)]
    #             j_h = self.v_h[(t, y, o)]
    #             j_p = self.v_p[(t, y, o)]
    #             j_k = k + self.v_k[(t, y, o)]
    #             self.j_b[(t, x, o)] = j_b
    #             self.j_h[(t, x, o)] = j_h
    #             self.j_p[(t, x, o)] = j_p
    #             self.j_k[(t, x, o)] = j_k

    #         return j_value

    def j_function_la(self, t, x, o):
        print((t, x, o))
        if (t, x, o) in self.value_function_j:
            return self.value_function_j[(t, x, o)]
        elif t == -1:
            return 0
        else:
            # Exploit S s structure
            q = self.order_la(t, x, o)
            y = x + q
            k = self.k if q > 0 else 0
            j_value = k + self.v_function_la(t, y, o)

            self.value_function_j[(t, x, o)] = j_value
            if self.detailed:
                j_b = self.v_b[(t, y, o)]
                j_h = self.v_h[(t, y, o)]
                j_p = self.v_p[(t, y, o)]
                j_k = k + self.v_k[(t, y, o)]
                self.j_b[(t, x, o)] = j_b
                self.j_h[(t, x, o)] = j_h
                self.j_p[(t, x, o)] = j_p
                self.j_k[(t, x, o)] = j_k

            return j_value

    def v_function_la(self, t, y, o):
        if (t, y, o) in self.value_function_v:
            return self.value_function_v[(t, y, o)]
        next_t, next_x, next_o = self.state_transition(t, y, o)
        new_states, probabilities = self.unpack_state_transition(next_t, next_x, next_o)
        value = self.G(y, o) + self.gamma * sum(p * self.j_function_la(*state)
                                                for p, state in zip(probabilities, new_states))

        if self.detailed:
            self.v_b[(t, y, o)] = self.G_b(y, o) + self.gamma * sum(p * self.j_function_b(*state)
                                                                    for p, state in zip(probabilities, new_states))
            self.v_h[(t, y, o)] = self.G_h(y, o) + self.gamma * sum(p * self.j_function_h(*state)
                                                                    for p, state in zip(probabilities, new_states))
            self.v_p[(t, y, o)] = self.G_p(y, o) + self.gamma * sum(p * self.j_function_p(*state)
                                                                    for p, state in zip(probabilities, new_states))
            self.v_k[(t, y, o)] = self.gamma * sum(p * self.j_function_k(*state)
                                                   for p, state in zip(probabilities, new_states))
        self.value_function_v[(t, y, o)] = value
        return value

    def j_function_db(self, t, x, o):
        if (t, x, o) in self.value_function_j:
            return self.value_function_j[(t, x, o)]
        elif t == -1:
            return 0
        else:
            # Exploit S s structure
            q = self.order_q_continuous(t, x, o)
            p_upper = q - int(q)
            p_lower = 1 - p_upper
            y_lower = x + int(q)
            y_upper = y_lower + 1
            k = self.k if q >= 1 else p_upper * self.k if q > 0 else 0
            j_value = k + p_upper * self.v_function_db(t, y_upper, o) + p_lower * self.v_function_db(t, y_lower, o)

            self.value_function_j[(t, x, o)] = j_value
            if self.detailed:
                j_b = p_upper * self.v_b[(t, y_upper, o)] + p_lower * self.v_b[(t, y_lower, o)]
                j_h = p_upper * self.v_h[(t, y_upper, o)] + p_lower * self.v_h[(t, y_lower, o)]
                j_p = p_upper * self.v_p[(t, y_upper, o)] + p_lower * self.v_p[(t, y_lower, o)]
                j_k = k + p_upper*self.v_k[(t, y_upper, o)] + p_lower*self.v_k[(t, y_lower, o)]
                self.j_b[(t, x, o)] = j_b
                self.j_h[(t, x, o)] = j_h
                self.j_p[(t, x, o)] = j_p
                self.j_k[(t, x, o)] = j_k
            return j_value

    def v_function_db(self, t, y, o):
        if (t, y, o) in self.value_function_v:
            return self.value_function_v[(t, y, o)]
        next_t, next_x, next_o = self.state_transition(t, y, o)
        new_states, probabilities = self.unpack_state_transition(next_t, next_x, next_o)
        value = self.G(y, o) + self.gamma * sum(p * self.j_function_la(*state)
                                                for p, state in zip(probabilities, new_states))

        if self.detailed:
            self.v_b[(t, y, o)] = self.G_b(y, o) + self.gamma * sum(p * self.j_function_b(*state)
                                                                    for p, state in zip(probabilities, new_states))
            self.v_h[(t, y, o)] = self.G_h(y, o) + self.gamma * sum(p * self.j_function_h(*state)
                                                                    for p, state in zip(probabilities, new_states))
            self.v_p[(t, y, o)] = self.G_p(y, o) + self.gamma * sum(p * self.j_function_p(*state)
                                                                    for p, state in zip(probabilities, new_states))
            self.v_k[(t, y, o)] = self.gamma * sum(p * self.j_function_k(*state)
                                                   for p, state in zip(probabilities, new_states))
        self.value_function_v[(t, y, o)] = value
        return value

    def compute_policy_la(self, args):
        t, x, o = args
        return self.order_la(t, x, o)

    def compute_policies_parallel_la(self, t, o=None):
        if o:
            args = list((t, x, o) for x in range(30))
        else:
            args = list((t, x, o) for x in range(30) for o in self.info_states())
        order_qs = Pool(os.cpu_count() - 1).map(self.compute_policy_la, args)
        for order_q, arg in zip(order_qs, args):
            self.order_la_cache[arg] = order_q

    def to_pickle(self, filename):
        with open(filename + "_model.pickle", "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def read_pickle(cls, filename):
        with open(filename, "rb") as f:
            m = pickle.load(f)
        return m
