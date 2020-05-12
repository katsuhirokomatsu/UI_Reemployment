import matplotlib.pyplot as plt
import json
import numpy as np


#-----------------------------------------------------------
# Counterfactual 1: optimal b_e
#-----------------------------------------------------------

with open("output/cf_work_incentives.json") as f:
	cf_result = json.load(f)

b_e_list = cf_result["b_e"]
cs_list = cf_result["cs"]
fe_list = cf_result["fe"]

fig, ax = plt.subplots(figsize = (16, 8))
ax.plot(b_e_list, cs_list, \
        label = "$CS_e$", \
		lw = 2.0, \
        color = "xkcd:lightish blue", \
        alpha = 0.8)

ax.plot(b_e_list, fe_list, \
        ls = "--", \
        label = "$FE_e$", \
		lw = 2.0, \
        color = "xkcd:lightish blue", \
        alpha = 0.8)

fontsize = 20
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
ax.set_xlabel("Work incentives $b_e$", fontsize = fontsize)
ax.set_ylabel("$CS_e \ FE_e$", fontsize = fontsize)
plt.legend(fontsize = fontsize)
plt.xticks(fontsize = fontsize)
plt.yticks(fontsize = fontsize)
plt.savefig("figure/counterfactual_b_e.png")



#------------------------------------------------------------
# Counterfactual 2: interaction of b_u & b_e
#------------------------------------------------------------

with open("output/cf_ui_benefits.json") as f:
	cf_result = json.load(f)

b_e_list = cf_result["b_e"]
b_u_list = cf_result["b_u"]
cs_list = np.array(cf_result["cs"])
fe_list = np.array(cf_result["fe"])


fig, ax = plt.subplots(figsize = (16, 8))

label_cs = ["$CS_u (b_e = 0.0)$", "$CS_u (b_e = 0.25)$", "$CS_u (b_e = 0.5)$"]
label_fe = ["$FE_u (b_e = 0.0)$", "$FE_u (b_e = 0.25)$", "$FE_u (b_e = 0.5)$"]
alpha = [0.3, 0.6, 1.0]

for i in range(3):
    l_cs, l_fe = label_cs[i], label_fe[i]

    ax.plot(b_u_list, cs_list[i, :], \
        label = l_cs, \
        lw = 2.0, \
        color = "xkcd:lightish blue", \
        alpha = alpha[i])

    ax.plot(b_u_list, fe_list[i, :], \
        ls = "--", \
        label = l_fe, \
        lw = 2.0, \
        color = "xkcd:lightish blue", \
        alpha = alpha[i])


fontsize = 20
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
ax.set_xlabel("UI replacement rate $b_u$", fontsize = fontsize)
ax.set_ylabel("$CS_u \ FE_u$", fontsize = fontsize)
plt.legend(ncol = 3, fontsize = fontsize)
plt.xticks(fontsize = fontsize)
plt.yticks(fontsize = fontsize)

plt.savefig("figure/counterfactual_b_u.png")