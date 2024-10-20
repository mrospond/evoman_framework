import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import single_ea

def save_best_weights_for_each_run() -> None:

    for run_folder in os.listdir("results2"):
        if run_folder == "plots":
            continue
        best_fit = 0
        best_weights = None
        with open(os.path.join("results2", run_folder, "weights_overall.csv")) as f:
            reader = csv.reader(f, delimiter=",")
            best_fit_folder = 0
            for line in reader:
                fit = float(line[0])
                if fit > best_fit:
                    best_fit = fit
                    weights = []
                    for w in line[1:]:
                        weights.append(float(w))
                    best_weights = weights
                    # print(f"\t{best_fit}")
                if fit > best_fit_folder:
                    best_fit_folder = fit
            # print(f"\t({best_fit_folder})")

        print("\nbest", best_fit, "in", run_folder)
        np.savetxt(os.path.join("results2", run_folder, "best_weights.txt"), best_weights)

def get_gain(weights: np.ndarray) -> np.float64:
    """
    calculates gain for given nn weights against all enemies
    """
    enemies = [x for x in range(1, 9)]

    ea = single_ea.SingleEA(
        "gains",
        "gains",
        enemies,
        1,
        np.array([weights])
    )

    player, enemy = ea.get_gain(weights)
    return player - enemy # gain

def save_boxplots_and_ttest(gains: dict, save_path: str) -> None:
    """
    generates two pairs of box plots (one pair per training group => 4 boxes)
    """
    gains_1347 = {}
    gains_2467 = {}

    for key, value in gains.items():
        if key.endswith("1347"):
            gains_1347[key] = value
        elif key.endswith("2467"):
            gains_2467[key] = value

    # plot for 2467
    data_2467 = [gains_2467["gen-2467"], gains_2467["spec-2467"]]
    plt.figure(figsize=(8, 6))
    plt.boxplot(data_2467, tick_labels=["gen-2467", "spec-2467"])
    plt.xticks(fontsize=16)
    plt.title("Box Plot of gen-2467 and spec-2467", fontsize=16)
    plt.ylabel("Gain", fontsize=16)
    plt.scatter([1] * len(gains_2467["gen-2467"]), gains_2467["gen-2467"], color='red', label='gen-2467', alpha=0.6)
    plt.scatter([2] * len(gains_2467["spec-2467"]), gains_2467["spec-2467"], color='blue', label='spec-2467', alpha=0.6)
    plt.savefig(os.path.join(save_path, 'boxplot_2467.png'))

    plt.clf()

    # plot for 1347
    data_1347 = [gains_1347["gen-1347"], gains_1347["spec-1347"]]
    plt.figure(figsize=(8, 6))
    plt.boxplot(data_1347, tick_labels=["gen-1347", "spec-1347"])
    plt.xticks(fontsize=16)
    plt.title("Box Plot of gen-1347 and spec-1347", fontsize=16)
    plt.ylabel("Gain", fontsize=16)
    plt.scatter([1] * len(gains_1347["gen-1347"]), gains_1347["gen-1347"], color='red', label='gen-1347', alpha=0.6)
    plt.scatter([2] * len(gains_1347["spec-1347"]), gains_1347["spec-1347"], color='blue', label='spec-1347', alpha=0.6)
    plt.savefig(os.path.join(save_path,'boxplot_1347.png'))

    ## TTESTS
    # t-tests between datasets
    from scipy import stats

    t_stat_2467, p_value_2467 = stats.ttest_ind(gains_2467["gen-2467"], gains_2467["spec-2467"])
    t_stat_1347, p_value_1347 = stats.ttest_ind(gains_1347["gen-1347"], gains_1347["spec-1347"])

    # results
    print("T-test for gen-2467 vs spec-2467:")
    print(f"T-statistic: {t_stat_2467}, P-value: {p_value_2467}")

    print("T-test for gen-1347 vs spec-1347:")
    print(f"T-statistic: {t_stat_1347}, P-value: {p_value_1347}")

    # significance level = 0.05
    if p_value_2467 < 0.05:
        print("significant difference between gen-2467 and spec-2467.")
    else:
        print("no significant difference between gen-2467 and spec-2467.")

    if p_value_1347 < 0.05:
        print("significant difference between gen-1347 and spec-1347.")
    else:
        print("no significant difference between gen-1347 and spec-1347.")


if __name__ == "__main__":
    save_best_weights_for_each_run()

    plot_save_path = "results2/plots/"
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)

    gains_dict = {}
    for run in os.listdir("results2"):
        if run == "plots":
            continue
        best_weights = np.loadtxt(os.path.join("results2", run, "best_weights.txt"))
        gain = get_gain(best_weights)
        prefix = run.split('_')[0]

        if prefix in gains_dict:
            gains_dict[prefix].append(gain)
        else:
            gains_dict[prefix] = [gain]

    # print(gains_dict)
    # for key, values in gains_dict.items():
    #     print(f"Prefix: {key}, Number of values: {len(values)}")

    save_boxplots_and_ttest(gains_dict, plot_save_path)