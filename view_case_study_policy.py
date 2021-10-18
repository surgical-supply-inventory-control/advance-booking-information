import pickle
from pprint import pprint
if __name__ == '__main__':
    item = "47320"
    backlogging_cost = 100
    lead_time = 1
    info = 2

    with open(f"model_artifact/case_study_policy/ns_policy_id_{item}_b_{backlogging_cost}_lt_{lead_time}_info_{info}.pickle", "rb") as f:
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
