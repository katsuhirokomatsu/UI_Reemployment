import numpy as np
import pandas as pd

# parameters
eta = 0.53
T_B = 26.0
D = 24.3
D_B = 15.8
gamma = 2

u = 0.054
u_B = D_B / T_B

def evaluate_formula(gamma, c_ue):
	mu = c_ue ** gamma
	ratio = u / (1 - u)
	out = ratio * D_B / D * mu * eta
	return out



def compute_MVPF(gamma, c_ue):
	mu = c_ue ** gamma
	ratio_B = u_B / (1 - u_B)
	FE = -ratio_B *  mu * eta 
	return 1 / (1 + FE)


data = []
for c_ue in [0.85, 0.9, 0.95]:
    gain1 = evaluate_formula(gamma = 1.0, c_ue = c_ue)
    gain2 = evaluate_formula(gamma = 2.0, c_ue = c_ue)
    gain3 = evaluate_formula(gamma = 3.0, c_ue = c_ue)
    data.append([c_ue,  round(gain1, 3), round(gain2, 3), round(gain3, 3)])

df = pd.DataFrame(data, columns = ["$c_u/c_e$", "$\gamma=1$", "$\gamma=2$", "$\gamma=3$"])


df.to_latex("table/welfare_impact.tex", index = False, escape = False)


data = []
for c_ue in [0.85, 0.9, 0.95]:
    gain1 = compute_MVPF(gamma = 1.0, c_ue = c_ue)
    gain2 = compute_MVPF(gamma = 2.0, c_ue = c_ue)
    gain3 = compute_MVPF(gamma = 3.0, c_ue = c_ue)
    data.append([c_ue,  round(gain1, 2), round(gain2, 2), round(gain3, 2)])

df = pd.DataFrame(data, columns = ["$c_u/c_e$", "$\gamma=1$", "$\gamma=2$", "$\gamma=3$"])
df.to_latex("table/MVPF.tex", index = False, escape = False)