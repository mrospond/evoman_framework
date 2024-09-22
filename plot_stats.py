import matplotlib.pyplot as plt
import pandas as pd

experiment_name = "decreasing_ea-4-100-20-0.1-0_005-15--2-2-2-020"
df = pd.read_csv(f"{experiment_name}/stats.csv", delimiter=",", dtype=str).astype(float)
enemy = experiment_name.split("-")[1]
pop_size = experiment_name.split("-")[2]

fig, (ax1, ax2) = plt.subplots(2)


df_fitness = df[["best_fit", "mean_fit", "std_fit"]]
if experiment_name.startswith("adaptative"):
    fig.suptitle(f"adaptative, enemy={enemy}")
    df_prob = df[["best_prob", "mean_prob", "std_prob"]]
else:
    fig.suptitle(f"decreasing static, enemy={enemy}, n={pop_size}")
    df_prob = df[["prob"]]

ax1.set_title("fitness")
ax2.set_title("mutation probability")

df_fitness.plot(ax=ax1)
df_prob.plot(ax=ax2)

doomsday_indices = df[df["doomsday"] == 1].index
for x in doomsday_indices:
    ax1.axvline(x=x, color="red", linewidth=0.5, linestyle="--", label="Doomsday")
    ax2.axvline(x=x, color="red", linewidth=0.5, linestyle='--', label="Doomsday")

plt.show()
