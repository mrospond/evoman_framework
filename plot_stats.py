import matplotlib.pyplot as plt
import pandas as pd


experiment_name = "specialized_ea"
df = pd.read_csv(f"{experiment_name}/stats.csv", delimiter=",", dtype=str).astype(float)
df_fitness = df[["best_fit", "mean_fit", "std_fit"]]
df_prob = df[["best_prob", "mean_prob", "std_prob"]]


fig, (ax1, ax2) = plt.subplots(2)
ax1.set_title("fitness")
ax2.set_title("mutation probability")

df_fitness.plot(ax=ax1)
df_prob.plot(ax=ax2)

plt.show()
