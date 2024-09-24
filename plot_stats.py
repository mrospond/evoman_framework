import matplotlib.pyplot as plt
import pandas as pd

experiment_name = "adaptative_ea-3-100-20-250-010-5--2-2-015"
df = pd.read_csv(f"{experiment_name}/stats.csv", delimiter=",", dtype=str).astype(float)
enemy = experiment_name.split("-")[1]
pop_size = experiment_name.split("-")[2]

fig, ax_left = plt.subplots()
ax_left.set_xlabel("iterations")
ax_left.set_ylabel("fitness")
lower = min(df["mean_fit"]) + 0.2*min(df["mean_fit"])
# ax_left.set_ylim([lower, 100])
ax_right = ax_left.twinx()
ax_right.set_ylabel("probability")

if experiment_name.startswith("adaptative"):
    type = "adaptative"
    df_prob = df[["mean_prob"]]
else:
    type = "decreasing"
    df_prob = df[["prob"]]

mean_fit_plot = df["mean_fit"].plot(ax=ax_left, legend=False, color='purple')
ax_left.fill_between(
    df.index,  # X-axis (shared index)
    df["mean_fit"] - df["std_fit"],
    df["mean_fit"] + df["std_fit"],
    color="purple", alpha=0.1,
)
df["best_fit"].plot(ax=ax_left, legend=False, color="blue")


ax_left.set_title(f"{type}, enemy={enemy}, n={pop_size}")
df_prob.plot(ax=ax_right, legend=False, color="green", linewidth=0.75)

doomsday_indices = df[df["doomsday"] == 1].index
i = 0
for x in doomsday_indices:
    if i == 0:
        ax_left.axvline(x=x, color="red", linewidth=0.5, linestyle="--", label="Doomsday")
    else:
        ax_left.axvline(x=x, color="red", linewidth=0.5, linestyle="--")
    i += 1

lines_left, labels_left = ax_left.get_legend_handles_labels()
lines_right, labels_right = ax_right.get_legend_handles_labels()

ax_left.legend(lines_right + lines_left, labels_right + labels_left, loc="upper right", bbox_to_anchor=(1.125, 0.975))

fig.tight_layout()
plt.show()
