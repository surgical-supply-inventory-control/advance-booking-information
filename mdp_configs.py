import sys, os, pacal,prepare
import scm_optimization.model as s_model
from scm_optimization.model import DeterministUsageModel, BinomUsageModel, PoissonUsageModel, GenPoissonUsageModel

if __name__ == "__main__":
    s_config = s_model.ModelConfig(
                            gamma=1,
                            lead_time=0,
                            holding_cost=1,
                            backlogging_cost=10,
                            setup_cost=0,
                            unit_price=1,
                            usage_model=GenPoissonUsageModel(scale=1,alpha=0),
                            alpha = 0,
                            horizon=1,
                            info_rv = pacal.BinomialDistr(10, 0.5),
                            label="ExperimentResults",
                            label_index=0)
    time_horizon = 21
    xs = list(range(1))
    ts = list(range(time_horizon))
    s_model.run_config(args = (s_config, ts, xs))
