import os
import matplotlib.pyplot as plt
import pandas as pd


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

def plot_result(stats: dict):

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
    
    plt.figure(figsize=(10, 6))
    plt.plot(generations, avg_best_fit, label='Average Best Fitness', color='blue', linestyle='-', marker='o')
    plt.fill_between(generations, avg_best_fit - std_best_fit, avg_best_fit + std_best_fit, color='blue', alpha=0.2)

    plt.plot(generations, avg_mean_fit, label='Average Mean Fitness', color='green', linestyle='--', marker='s')
    plt.fill_between(generations, avg_mean_fit - std_mean_fit, avg_mean_fit + std_mean_fit, color='green', alpha=0.2)

    plt.plot(generations, avg_median_prob, label='Average Median Probability', color='red', linestyle='-.', marker='^')
    plt.fill_between(generations, avg_median_prob - std_median_prob, avg_median_prob + std_median_prob, color='red', alpha=0.2)

    
    plt.title('Fitness and mutation probability in each generation')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(True)

    plot_name = os.path.basename(next(iter(stats)))[:-2]
    plt.savefig(os.path.join(save_path, plot_name))
    print("saved " + plot_name)



def box_plot():
    pass

def best_fitness():
    pass

if __name__ == "__main__":
    dir_adaptative = "results/adaptative_ea_bad"
    dir_decreasing = "results/decreasing_ea"
    
    ENEMY=7

    stats_adaptative=read_stats(dir_adaptative, f"{os.path.basename(dir_adaptative)}-{ENEMY}")
    # print(stats_adaptative.keys())

    plot_result(stats_adaptative)
