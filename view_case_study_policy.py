import pickle
from pprint import pprint
if __name__ == '__main__':
    item = "ItemA"
    backlogging_cost = 100
    lead_time = 2
    info = 2

    with open("model_artifact/case_study_policy/ns_policy_id_ItemA_b_100_lt_0_info_2.pickle", "rb") as f:
        policy = pickle.load(f)

    days = {0: "Monday",
            1: "Tuesday",
            2: "Wednesday",
            3: "Thursday",
            4: "Friday",
            5: "Saturday",
            6: "Sunday",
            }

    print("Policy read as (info state): (BaseStockLevel, ReorderPoint)")
    for weekday in range(len(policy)):
        print("Ordering policy on ", days[weekday])
        pprint(policy[weekday])
