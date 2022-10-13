import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

NAME = "CartPole-v0_NonMdpHC_NonMdp"  # or "CartPole-v0_PPO"
df_cart = pd.read_pickle(f"../results/{NAME}_CART/database.pk")
df_opct = pd.read_pickle(f"../results/{NAME}_OPCT/database.pk")
plt.style.use('seaborn-darkgrid')

depths = df_cart.max_depth.unique()

o = df_cart[df_cart["max_depth"] == depths[-1]]
mo = np.mean(o['oracle_mean'])
so = np.mean(o['oracle_std'])
print("Oracle average mean and average standard deviation is: "
      f"<R> = {mo} +/- {so}")

ms_cart = np.zeros(len(depths))
ss_cart = np.zeros(len(depths))
ms_opct = np.zeros(len(depths))
ss_opct = np.zeros(len(depths))
for i, depth in enumerate(depths):
    ms_cart[i] = np.mean(df_cart[df_cart["max_depth"] == depth]["mean"])
    ss_cart[i] = np.mean(df_cart[df_cart["max_depth"] == depth]["std"])
    print(f"At depth={depth} the average mean and average standard deviation "
          f"for CART is: <R> = {ms_cart[i]} +/- {ss_cart[i]}")
    ms_opct[i] = np.mean(df_opct[df_opct["max_depth"] == depth]["mean"])
    ss_opct[i] = np.mean(df_opct[df_opct["max_depth"] == depth]["std"])
    print(f"At depth={depth} the average mean and average standard deviation "
          f"for OPCT is: <R> = {ms_opct[i]} +/- {ss_opct[i]}")


color_cart = sns.color_palette("deep")[0]
color_opct = sns.color_palette("deep")[1]
fig, axs = plt.subplots(1, 1, sharey=True)

if NAME.split("_")[-1] == "NonMdp":
    oracle_label = "Oracle (HC)"
elif NAME.split("_")[-1] == "PPO":
    oracle_label = "Oracle (PPO)"
else:
    print("Could not determine oracle from filename")
    raise ValueError
axs.plot(depths, mo*np.ones(len(depths)), c='k', alpha=0.5, linewidth=1,
         linestyle="--", label=oracle_label)
axs.fill_between(depths, mo - so, mo + so, color='k', alpha=0.1)

axs.plot(depths, ms_cart, c=color_cart, label="CART")
axs.fill_between(depths, ms_cart - ss_cart, ms_cart + ss_cart,
                 color=color_cart, alpha=0.2)
axs.plot(depths, ms_opct, c=color_opct, label="OPCT")
axs.fill_between(depths, ms_opct - ss_opct, ms_opct + ss_opct,
                 color=color_opct, alpha=0.2)

fs = 16
axs.set_xlabel("depth", fontsize=fs+2)
axs.set_xticks(df_cart["max_depth"].unique())
axs.set_ylabel(r"$\overline{R}$", fontsize=fs+2)
plt.ylim([25, 210])
axs.tick_params(axis='both', which='major', labelsize=fs)
plt.legend(loc="lower right", fontsize=fs)
plt.savefig(NAME + ".pdf", bbox_inches="tight")
plt.close()

# Print one CART and one OPCT as example
seed = 16
d = 3
print(df_cart[(df_cart["seed"] == seed+1) &
              (df_cart["max_depth"] == d)]["tree"].iloc[0])
df_opct[(df_opct["seed"] == seed+1) &
        (df_opct["max_depth"] == d)]["tree"].iloc[0].print_tree_pythonlike()
