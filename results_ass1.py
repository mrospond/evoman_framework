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


def plot_stats(stats: dict, enemy: int):

    print("TEST")

    save_path = "results/plots/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    all_best_fit=[]
    all_mean_fit=[]
    all_median_prob=[]

    for _, df in stats.items():
        all_best_fit.append(df['best_fit'])
        all_mean_fit.append(df['mean_fit'])
        all_median_prob.append(df['median_prob'] * 100)
    
    best_fit_df = pd.DataFrame(all_best_fit)
    mean_fit_df = pd.DataFrame(all_mean_fit)
    median_prob_df = pd.DataFrame(all_median_prob)

    avg_best_fit = best_fit_df.mean(axis=0)
    avg_mean_fit = mean_fit_df.mean(axis=0)
    avg_median_prob = median_prob_df.mean(axis=0)

    std_best_fit = best_fit_df.std(axis=0)
    std_mean_fit = mean_fit_df.std(axis=0)
    std_median_prob = median_prob_df.std(axis=0)

    generations = range(len(avg_best_fit))  # all experiments should have the same number of generations
    
    def plot_plot():
        plt.figure(figsize=(10, 6))
        plt.plot(generations, avg_best_fit, label='Average Best Fitness', color='blue', linestyle='-',)
        plt.fill_between(generations, avg_best_fit - std_best_fit, avg_best_fit + std_best_fit, color='blue', alpha=0.2)
        plt.plot(generations, avg_mean_fit, label='Average Mean Fitness', color='green', linestyle='-')
        plt.fill_between(generations, avg_mean_fit - std_mean_fit, avg_mean_fit + std_mean_fit, color='green', alpha=0.2)
        plt.plot(generations, avg_median_prob, label='Average Median Probability', color='red', linestyle='-')
        plt.fill_between(generations, avg_median_prob - std_median_prob, avg_median_prob + std_median_prob, color='red', alpha=0.2)
        

        plt.title(f'Average fitness in 10 runs of EA1 for Enemy {enemy}')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.legend(
            loc='lower left', bbox_to_anchor=(0.5, 0.2),
            fontsize=12,
            markerscale=2,
            # handlelength=3,
            # handleheight=2, #legend
            borderpad=1.5, # Padding between the legend border and content
            # frameon=False,
            # shadow=True,
        )
        plt.grid(True)
        plot_name = os.path.basename(next(iter(stats)))[:-2]
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
        'Weights': best_weights
    }

    # Output the best fitness information to a file
    with open(save_path, 'w') as file:
        for weight in best_weights:
            file.write(f"{weight}\n")
    
    print(f"Best fitness from generation: {best_fitness_gen}, valued: {best_fitness_value} weights saved to {save_path}")

    return best_info

def calculate_gain(weights_file_path: str, enemies: list) -> dict:
    """
    read weights from weights_file_path, then run 5 experiments and calculate gain
    return list with gain for n runs as well as corresponding enemy
    """
    runs = 5

    # if not os.path.exists(experiment_name):
    #     raise Exception(experiment_name + " file doesnt exist!")

    #get dir with best weight

    with open(weights_file_path, "r") as f:
        lines = f.readlines()
        best_ind = np.array([float(line.strip()) for line in lines if line.strip()])

    all_gains_dict = {}
    for enemy in enemies:
        gains_list = []
        experiment_name="test_experiment"
        if not os.path.exists(experiment_name):
            os.makedirs(experiment_name)

        ea = adaptative_ea_bad.SpecializedEA(experiment_name, enemy)

        for _ in range(runs):
            fitness, playerlife, enemylife, time = ea.env.play(pcont=best_ind)
            gains_list.append(playerlife - enemylife)
            print(ea.env.play(pcont=best_ind))

            
            all_gains_dict[f"{weights_file_path.split('/')[1].replace('_ea', '').replace('_bad', '')} E{enemy}"] = gains_list
    

    return all_gains_dict


def save_box_plot(all_gains_dict: dict):
    """
    @params - dict with all gains for 5 runs for each enemy and ea
    """

    gains = list(all_gains_dict.values())
    labels = [f"{key}" for key in all_gains_dict.keys()]

    colors = ['coral' if i%2==0 else 'crimson' for i in range(len(labels))]

    print(len(colors))

    _, ax = plt.subplots()
    ax.set_ylabel('Gain')

    bplot = ax.boxplot(gains,
                    patch_artist=True,
                    tick_labels=labels,
                    medianprops=dict(color="black"))

    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    plt.savefig("results/plots/box-plot")
    print("saved box_plot")


if __name__ == "__main__":
    
    dir_adaptative = "results/adaptative_ea_bad"
    dir_decreasing = "results/decreasing_ea"
    enemies=[7, 8]

   

    for enemy in enemies:
         # where to save weights
        experiment_name_adaptative = f"{dir_adaptative}/fittest_weights_{enemy}"
        experiment_name_decreasing = f"{dir_decreasing}/fittest_weights_{enemy}"
        print(f"RUNNING FOR ENEMY {enemy}")
        stats_adaptative = read_stats(dir_adaptative, f"{os.path.basename(dir_adaptative)}-{enemy}")
        stats_decreasing = read_stats(dir_decreasing, f"{os.path.basename(dir_decreasing)}-{enemy}")
        print("OK")

        # plot_stats(stats_adaptative, enemy)
        plot_stats(stats_decreasing, enemy)
        print("OK")

        save_best_individual(stats_adaptative, experiment_name_adaptative)
        save_best_individual(stats_decreasing, experiment_name_decreasing)
        print("OK")




#####
    gains_dict_adaptative = calculate_gain(experiment_name_adaptative, enemies)
    gains_dict_decreasing = calculate_gain(experiment_name_decreasing, enemies)

    all_gains_dict = gains_dict_adaptative | gains_dict_decreasing

    print(all_gains_dict.keys())

    print(all_gains_dict)
    print("OK")
    save_box_plot(all_gains_dict)
 