# advance-booking-information
Demo code on studying the value of advanced information in management of surgical supplies in hospitals.

Four demo scripts have been prepared to demonstrate this project.

1. run_basic_mdp_experiment.py
2. run_case_study_mdp.py
3. run_case_study_simulation.py
4. view_case_study_policy.py

The run_basic_mdp_experiment script shows how to run one of the basic experiments with Poisson item 
usage and Binomial booking distribution. The number of periods of ABI used, leadtime and costs 
can be easily changed as desired.

The run run_case_study_mdp script shows how to run the mdp to get a policy for one of the 5 items
in the case study. Note that as the raw data can not be shared, the script loads the 
pre-constructed information process represented by the info_state_rvs attribute of the model class 
from a pickle file.

The run_case_study_simulation script shows how to run the case study simulation for one of the 5
items in the case study. As solving the backward induction is time consuming and the model files are
very large after solving, the model's policy is stored as a pickle file and can be viewed using the 
view_case_study_policy script. 