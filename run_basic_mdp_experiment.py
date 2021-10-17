from scm_optimization.model import ModelConfig, run_config, PoissonUsageModel
import pacal
import pandas as pd

if __name__ == "__main__":
    xs = list(range(0, 1))
    ts = list(range(0, 21))

    model = run_config(
        [
            ModelConfig(
                gamma=1,
                lead_time=0,
                info_state_rvs=None,
                holding_cost=1,
                backlogging_cost=100,
                setup_cost=0,
                unit_price=0,
                usage_model=PoissonUsageModel(1),
                horizon=1,
                info_rv=pacal.BinomialDistr(10, 0.5),
                label="base_experiment_detailed",
                detailed=True,
                label_index=0),
            ts,
            xs
        ]
    )

    cost_results = pd.DataFrame()
    t = max(ts)
    print("Cost Function at last time step starting with no inventory: t={}, x=0".format(str(t)))
    print("(info_state): cost")
    for o in model.info_states():
        cost_results = cost_results.append(
            {"info_state": o,
             "cost": model.value_function_j[(t, 0, o)]
             },
            ignore_index=True
        )
    print(cost_results[["info_state", "cost"]])

    base_stock_results = pd.DataFrame()
    print("Order Policy at last time step: t={}, x=0".format(str(t)))
    print("(info_state): cost")
    for o in model.info_states():
        base_stock_results = base_stock_results.append(
            {"info_state": o,
             "base_stock_level": model.stock_up_level(t, o),
             "reorder_point": model.base_stock_level(t, o)
             },
            ignore_index=True
        )
    print(base_stock_results[["info_state", "base_stock_level", "reorder_point"]])
