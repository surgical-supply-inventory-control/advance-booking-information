import scm_optimization.non_stationary_model as ns_model
from scm_optimization.model import PoissonUsageModel
from model_artifact.ns_info_state_rvs.rv import elective_info_rvs, emergency_info_rvs

import pacal

configs = []
item_ids = [id for id in elective_info_rvs]

if __name__ == "__main__":
    item = "ItemB"
    lead_time = 1
    b = 100

    xs = list(range(1))
    ts = list(range(7 * 4))

    model = ns_model.run_config(
        [
            ns_model.ModelConfig(
                gamma=1,
                lead_time=1,
                holding_cost=1,
                backlogging_cost=b,
                setup_cost=0,
                unit_price=0,
                usage_model=PoissonUsageModel(scale=1),
                elective_info_state_rv=elective_info_rvs[item],
                emergency_info_state_rv=emergency_info_rvs[item],
                horizon=2,
                label="ns_impl_b_{}_LT_{}_{}".format(str(b), str(lead_time), item),
                label_index=0),
            ts,
            xs
        ]
    )
