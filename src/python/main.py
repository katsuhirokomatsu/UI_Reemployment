import ui_simulation as ui
import numpy as np
import json
import sys

with open("input/model_params.json") as f:
	model_params = json.load(f)

with open("input/grid_params.json") as f:
	grid_params = json.load(f)

with open("input/moments.json") as f:
	moments = json.load(f)

m = ui.Model(model_params, grid_params)

# Calibration

print("Estimate? 1: Yes, 0: No")
est = int(input())
if est == 1:
	params = ui.estimate_parameters(m, moments)
	est_result = {}
	est_result["xi"] = params[0]
	est_result["theta"] = params[1]
	with open("output/estimation_result.json", "w") as f:
		json.dump(est_result, f)

elif est == 0:
	with open("output/estimation_result.json") as f:
		params = json.load(f)
else:
	sys.exit("Input 1 or 0")

m.set_calibrated_params(params)


# Fit
sol = ui.Solution(m)
sol.solve()
print(sol.compute_duration())
print(sol.compute_duration_elasticity())
print(sol.compute_consumption_drop(t_end = 26))



# Counterfactual
N = 10
b_u = 0.5

cf = ui.Counterfactual(m)

## change b_e
b_e_list = np.linspace(0.1, 0.9, N)
cs_list = [None] * N
fe_list = [None] * N

for i, b_e in enumerate(b_e_list):
	cs, fe = cf.compute_welfare_effect_employed(b_u, b_e)
	cs_list[i] = cs
	fe_list[i] = fe

cf_result = {}
cf_result["b_e"] = b_e_list.tolist()
cf_result["cs"] = cs_list
cf_result["fe"] = fe_list

with open("output/cf_work_incentives.json", "w") as f:
	json.dump(cf_result, f)




## change b_u
b_e_list = [0.0, 0.25, 0.5]
b_u_list = np.linspace(0.1, 0.6, N)
cs_list = np.empty([3, N])
fe_list = np.empty([3, N])

for i, b_e in enumerate(b_e_list):
	for j, b_u in enumerate(b_u_list):
		cs, fe = cf.compute_welfare_effect_unemployed(b_u, b_e)
		cs_list[i, j] = cs
		fe_list[i, j] = fe

cf_result = {}
cf_result["b_e"] = b_e_list
cf_result["b_u"] = b_u_list.tolist()
cf_result["cs"] = cs_list.tolist()
cf_result["fe"] = fe_list.tolist()

with open("output/cf_ui_benefits.json", "w") as f:
	json.dump(cf_result, f, indent = 4)