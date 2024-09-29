import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import adaptative_ea_bad
import decreasing_ea

def read_stats(base_dir: str, prefix: str) -> dict:
    stats = {}

    for subdir, _, files in os.walk(base_dir):
        if os.path.basename(subdir).startswith(prefix): # look only for runs against one enemy
            for file in files:
                if file == "stats.csv":
                    file_path = os.path.join(subdir, file)

                    try:
                        df = pd.read_csv(file_path).astype(float)
                        stats[subdir] = df #{exp_name: stats_df}
                    except Exception as e:
                        print(e)
    return stats

def plot_stats(stats_adaptative: dict, stats_decreasing: dict, enemy: int):
    
    save_path = "results/plots/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # adaptive EA data
    all_best_fit_ad = []
    all_mean_fit_ad = []
    all_median_prob_ad = []

    for _, df in stats_adaptative.items():
        all_best_fit_ad.append(df['best_fit'])
        all_mean_fit_ad.append(df['mean_fit'])
        if 'median_prob' in df.columns:
            all_median_prob_ad.append(df['median_prob'] * 100)
        elif 'prob' in df.columns:
            all_median_prob_ad.append(df['prob'] * 100)
        else:
            raise Exception("whoops")

    best_fit_df_ad = pd.DataFrame(all_best_fit_ad)
    mean_fit_df_ad = pd.DataFrame(all_mean_fit_ad)
    median_prob_df_ad = pd.DataFrame(all_median_prob_ad)

    avg_best_fit_ad = best_fit_df_ad.mean(axis=0)
    avg_mean_fit_ad = mean_fit_df_ad.mean(axis=0)
    avg_median_prob_ad = median_prob_df_ad.mean(axis=0)

    std_best_fit_ad = best_fit_df_ad.std(axis=0)
    std_mean_fit_ad = mean_fit_df_ad.std(axis=0)
    std_median_prob_ad = median_prob_df_ad.std(axis=0)
    generations = range(len(avg_best_fit_ad))

    # decreasing EA data
    all_best_fit_de = []
    all_mean_fit_de = []
    all_median_prob_de = []

    for _, df in stats_decreasing.items():
        all_best_fit_de.append(df['best_fit'])
        all_mean_fit_de.append(df['mean_fit'])
        if 'median_prob' in df.columns:
            all_median_prob_de.append(df['median_prob'] * 100)
        elif 'prob' in df.columns:
            all_median_prob_de.append(df['prob'] * 100)
        else:
            raise Exception("whoops")

    best_fit_df_de = pd.DataFrame(all_best_fit_de)
    mean_fit_df_de = pd.DataFrame(all_mean_fit_de)
    median_prob_df_de = pd.DataFrame(all_median_prob_de)

    avg_best_fit_de = best_fit_df_de.mean(axis=0)
    avg_mean_fit_de = mean_fit_df_de.mean(axis=0)
    avg_median_prob_de = median_prob_df_de.mean(axis=0)

    std_best_fit_de = best_fit_df_de.std(axis=0)
    std_mean_fit_de = mean_fit_df_de.std(axis=0)
    std_median_prob_de = median_prob_df_de.std(axis=0)

    def plot_plot():
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()
        ax1.set_xlabel('Generation', fontsize=16)
        ax1.set_ylabel('Fitness', fontsize=16)
        ax2.set_ylabel("Mutation probability (%)", fontsize=16)
        fig.suptitle(f'Average fitness in a EA run for E{enemy}', fontsize=18)

        # adaptative
        ax1.plot(generations, avg_best_fit_ad, label='best fitness adaptive', color='blue', lw=1)
        ax1.fill_between(generations, avg_best_fit_ad - std_best_fit_ad, avg_best_fit_ad + std_best_fit_ad, color='blue', alpha=0.2)
        ax1.plot(generations, avg_mean_fit_ad, label='mean fitness adaptive', color='green', lw=1)
        ax1.fill_between(generations, avg_mean_fit_ad - std_mean_fit_ad, avg_mean_fit_ad + std_mean_fit_ad, color='green', alpha=0.2)
        ax2.plot(generations, avg_median_prob_ad, label='median probability Adaptive', color='red', lw=1)
        ax2.fill_between(generations, avg_median_prob_ad - std_median_prob_ad, avg_median_prob_ad + std_median_prob_ad, color='red', alpha=0.2)

        # decreasing
        ax1.plot(generations, avg_best_fit_de, label='best fitness decreasing', color='blue', lw=1, linestyle='--')
        ax1.fill_between(generations, avg_best_fit_de - std_best_fit_de, avg_best_fit_de + std_best_fit_de, color='blue', alpha=0.2)
        ax1.plot(generations, avg_mean_fit_de, label='mean fitness decreasing', color='green', lw=1, linestyle='--')
        ax1.fill_between(generations, avg_mean_fit_de - std_mean_fit_de, avg_mean_fit_de + std_mean_fit_de, color='green', alpha=0.2)
        ax2.plot(generations, avg_median_prob_de, label='median probability decreasing', color='red', lw=1, linestyle='--')
        ax2.fill_between(generations, avg_median_prob_de - std_median_prob_de, avg_median_prob_de + std_median_prob_de, color='red', alpha=0.2)

        fig.autofmt_xdate()
        fig.legend(
            loc='lower left', bbox_to_anchor=(0.56, 0.255),
            fontsize=13,
            markerscale=2,
            framealpha=1
        )

        ax1.grid(True)
        ax2.set_ylim(ax1.get_ylim())
        fig.tight_layout()

        plot_name = f"enemy-{enemy}"
        plt.savefig(os.path.join(save_path, plot_name))
        print("saved " + plot_name)

    # plotting
    plot_plot()


def save_best_individual(stats: dict, save_path: str):
    """
    @param dict - {dir: stats_df}
    @param save_path - where to save weights for best fitness individual
    @param enemy - enemy
    @return best info (optional)
    """

    best_fitness_value = -10000
    best_fitness_gen = None
    best_fitness_subdir = None

    for run, df in stats.items():
        current_best_value = df['best_fit'].max()
        current_best_gen = df['best_fit'].idxmax()  # generation number of the best value

        if current_best_value > best_fitness_value:
            best_fitness_value = current_best_value
            best_fitness_gen = current_best_gen
            best_fitness_subdir = str(run)

    weights_file_path = os.path.join(best_fitness_subdir, "weights.csv")
    try:
        with open(weights_file_path, 'r') as file:
            weights_lines = file.readlines()
            best_weights = [float(w) for w in weights_lines[best_fitness_gen].strip().split(',')]
    except Exception as e:
        print(e)
        best_weights = None

    # Save the best fitness information
    best_info = {
        'Best Fitness Value': best_fitness_value,
        'Generation Number': best_fitness_gen,
        'Subdirectory': best_fitness_subdir,
        # 'Weights': best_weights
    }

    # Output the best fitness information to a file
    with open(save_path, 'w') as file:
        for weight in best_weights:
            file.write(f"{weight}\n")

    print(f"Best fitness from generation: {best_fitness_gen}, valued: {best_fitness_value} weights saved to {save_path}")

    print (best_info)

def calculate_gain(ea_kind: str, enemies: list) -> dict:
    """
    read weights from weights_file_path, then run 5 experiments and calculate gain
    return list with gain for n runs as well as corresponding enemy
    """
    runs = 5


    experiment_name="test_experiment"
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)


    all_gains_dict = {}
    for enemy in enemies:
        weights_file_path = f"{ea_kind}/fittest_weights_{enemy}"
        with open(weights_file_path, "r") as f:
            lines = f.readlines()
            best_ind = np.array([float(line.strip()) for line in lines if line.strip()])

        gains_list = []
        # print(weights_file_path.split('/')[1], "\n\n\nn\n")

        if weights_file_path.split('/')[1] == "adaptative_ea_bad":
            ea = adaptative_ea_bad.SpecializedEA(experiment_name, enemy)
        else:
            ea = decreasing_ea.SpecializedEA(experiment_name, enemy)
        print(f"{ea_kind}, enemy: {enemy}\n",)
        for _ in range(runs):
            fitness, playerlife, enemylife, time = ea.env.play(pcont=best_ind)
            gains_list.append(playerlife - enemylife)
            print(ea.env.play(pcont=best_ind))

        all_gains_dict[f"{weights_file_path.split('/')[1].replace('_ea', '').replace('_bad', '').replace('adaptative','adaptive')} E{enemy}"] = gains_list
        # ea=''


    return all_gains_dict


def save_box_plot(all_gains_dict: dict):
    """
    @params - dict with all gains for 5 runs for each enemy and ea
    """

    gains = list(all_gains_dict.values())
    labels = [f"{key}" for key in all_gains_dict.keys()]

    colors = ['coral' if i%2==0 else 'crimson' for i in range(len(labels))]

    print(len(colors))

    fig, ax = plt.subplots()
    ax.set_ylabel('Individual gain', fontsize=16)
    ax.set_xlabel('Experiment name', fontsize=16)
    ax.set_title('Mean individual gain of best performing\nindividual in adaptive and decreasing EA', fontsize=16)


    bplot = ax.boxplot(gains,
                    patch_artist=True,
                    tick_labels=labels,
                    medianprops=dict(color="black"))

    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    plt.tight_layout()
    fig.autofmt_xdate()


    plt.savefig("results/plots/box-plot")
    print("saved box_plot")


if __name__ == "__main__":

    dir_adaptative = "results/adaptative_ea_bad"
    dir_decreasing = "results/decreasing_ea"
    enemies=[2, 7, 8]



    for enemy in enemies:
         # where to save weights
        experiment_name_adaptative = f"{dir_adaptative}/fittest_weights_{enemy}"
        experiment_name_decreasing = f"{dir_decreasing}/fittest_weights_{enemy}"
        print(f"RUNNING FOR ENEMY {enemy}")
        stats_adaptative = read_stats(dir_adaptative, f"{os.path.basename(dir_adaptative)}-{enemy}")
        stats_decreasing = read_stats(dir_decreasing, f"{os.path.basename(dir_decreasing)}-{enemy}")
        print("OK")

        plot_stats(stats_adaptative, stats_decreasing, enemy)
        print("OK")

        save_best_individual(stats_adaptative, experiment_name_adaptative)
        save_best_individual(stats_decreasing, experiment_name_decreasing)
        print("OK")


#####
    gains_dict_adaptative = calculate_gain(dir_adaptative, enemies)
    gains_dict_decreasing = calculate_gain(dir_decreasing, enemies)

    all_gains_dict = gains_dict_adaptative | gains_dict_decreasing

    print(all_gains_dict.keys())
    print(all_gains_dict.items())

    print(all_gains_dict)
    print("OK")
    save_box_plot(dict(sorted(all_gains_dict.items(), key=lambda x: int(x[0].split('E')[-1]))))
