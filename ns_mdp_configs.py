import sys, os, pacal, prepare
import scm_optimization.non_stationary_model as ns_model
from scm_optimization.model import DeterministUsageModel, BinomUsageModel, PoissonUsageModel, GenPoissonUsageModel
from scm_implementation.ns_info_state_rvs.ns_info_state_rvs import elective_info_rvs, emergency_info_rvs,alphas

if __name__ == "__main__":
    ns_config = ns_model.ModelConfig(
                        gamma=1,
                        lead_time=0,
                        holding_cost=1,
                        backlogging_cost=10,
                        setup_cost=0,
                        unit_price=0,
                        usage_model=GenPoissonUsageModel(scale=1,alpha=0),
                        elective_info_state_rv= pacal.BinomialDistr(10, 0.5), #elective_info_rvs[item]
                        emergency_info_state_rv= pacal.BinomialDistr(3, 0.8), #emergency_info_rvs[item]
                        alpha = 0, #alphas[item]
                        horizon= 1,
                        label='ExperimentResults',
                        label_index=0)
    time_horizon = 28
    xs = list(range(1))
    ts = list(range(time_horizon))
    ns_model.run_config((ns_config, ts, xs))