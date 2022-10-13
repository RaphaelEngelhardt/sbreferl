import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

NAME = "MountainCarContinuous-v0_HC"  # or "MountainCarContinuous-v0_TD3"
ORACLE = NAME.split("_")[-1]
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

axs.plot(depths, mo*np.ones(len(depths)), c='k', alpha=0.5, linewidth=1,
         linestyle="--", label=f"Oracle ({ORACLE})")
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
plt.ylim([10, 110])
axs.tick_params(axis='both', which='major', labelsize=fs)
plt.legend(loc="lower right", fontsize=fs)
plt.savefig(NAME + ".pdf", bbox_inches="tight")
plt.close()

# Plot samples
plt.style.use('seaborn-whitegrid')
seed = 16
d = 3
o_samples = df_cart[(df_cart["seed"] == seed) &
                    (df_cart["max_depth"] == depths[-1])]["oracle_samples"].iloc[0]
cart_samples = df_cart[(df_cart["seed"] == seed+1) &
                     (df_cart["max_depth"] == d)]["tree_samples"].iloc[0]
opct_samples = df_opct[(df_opct["seed"] == seed+1) &
                       (df_opct["max_depth"] == d)]["tree_samples"].iloc[0]

fig, axs = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(10, 5))
sns.scatterplot(data=o_samples, x="car_pos", y="car_vel", hue="Decision",
                ax=axs[0])
sns.scatterplot(data=cart_samples, x="car_pos", y="car_vel", hue="Decision",
                ax=axs[1])
sns.scatterplot(data=opct_samples, x="car_pos", y="car_vel", hue="Decision",
                ax=axs[2])

if ORACLE == "TD3":
    axs[0].set_title("Oracle (TD3)")
elif ORACLE == "HC":
    axs[0].set_title("Oracle (handcrafted)")
else:
    print("Could not determine oracle from filename")
    raise ValueError
axs[1].set_title(f"CART depth {d}")
axs[2].set_title(f"OPCT depth {d}")
for i in range(3):
    axs[i].set_xlabel("x")
    axs[i].set_ylabel("v")
    axs[i].legend().set_visible(False)
    axs[i].set_aspect(1.0/axs[i].get_data_ratio(), adjustable='box')

handles, labels = axs[-1].get_legend_handles_labels()
fig.legend(handles, labels, loc='center right', ncol=1, title="Action")
plt.savefig("comp_" + NAME + ".pdf", bbox_inches="tight")
plt.close()

# Print one CART and one OPCT as example
print(df_cart[(df_cart["seed"] == seed+1) &
              (df_cart["max_depth"] == d)]["tree"].iloc[0])
df_opct[(df_opct["seed"] == seed+1) &
        (df_opct["max_depth"] == d)]["tree"].iloc[0].print_tree_pythonlike()
