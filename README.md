# Steps for Setup
In order to run the scripts, the working directory should be properly specified. This can be done by updating the directory variable in prepare.py
 
# Solving the MDP
	# Stationary Demand
		- Under stationary surgery distribution specify parameters in mdp_configs.py 
		- Run mdp_configs.py script python mdp_configs.py
	# Nonstationary Demand
		- Under nonstationary surgery distribution specify parameters in ns_mdp_configs.py 
		- Run mdp_configs.py script python ns_mdp_configs.py
		- The resulting policy and value functions is saved under the provided label

# Running Simulations
	# Evaluting specific policies
		- Specify the policies, cost parameters, item demand distribution and surgery distributions in run_simulation.py
		- Run run_simulation.py script python run_simulation.py
		- If policy, item demand distributions and surgery distributions are stored in a pickle file:
			- Specify the directories in the proper locations in run_empirical_case_study_simulation.py
			- Run run_empirical_case_study_simulation.py script python run_empirical_case_study_simulation.py
	# Running Approximation Algorithms: Dual Balancing (DB) and Look Ahead (LA)
		# Look Ahead (LA)
			- Specify the policy parameters in dual_balancing_extension/run_la_sim_args.py
			- Run dual_balancing_extension/run_la_sim_args.py script python dual_balancing_extension/run_la_sim_args.py
		# Dual Balancing (DB)
			- Specify the policy parameters in dual_balancing_extension/run_db_sim_args.py
			- Run dual_balancing_extension/run_db_sim_args.py script python dual_balancing_extension/run_db_sim_args.py
		
# Exact Evaluation of Approximation Algorithms: Dual Balancing (DB) and Look Ahead (LA)
	- Source code for the exact evaluation is provided under scm_optimization/heuristic_models
	# Dual Balancing (DB)
		- Specify the parameters in dual_balancing_extension/run_db_exact.py
		- Run run_db_exact.py script python dual_balancing_extension/run_db_exact.py
		- Resulting objective is printed at the end of the evaluation
	# Look Ahead (LA)
		- Specify the parameters in dual_balancing_extension/run_la_exact.py
		- Run run_db_exact.py script python dual_balancing_extension/run_la_exact.py
		- Resulting objective is printed at the end of the evaluation
