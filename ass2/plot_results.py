import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from global_params import *


def read_stats_overall(base_dir: str, prefix: str) -> dict:
    """
    reads stats_overall.csv file from the run with a given prefix
    returns dict{run_name: stats_df}
    """
    stats = {}

    for subdir, _, files in os.walk(base_dir):
        if os.path.basename(subdir).startswith(prefix): # look only for runs against one enemy
            for f in files:
                if f == "stats_overall.csv":
                    file_path = os.path.join(subdir, f)

                    try:
                        df = pd.read_csv(file_path).astype(float)
                        stats[subdir] = df #{exp_name: stats_df}
                    except Exception as e:
                        print(f, e)
    return stats

def read_stats_unprocessed(base_dir: str, prefix: str) -> dict:
    """
    reads stats.csv file from the run with a given prefix
    average fitness across duplicated generations
    returns dict{run_name: stats_df} with averaged data per generation
    """
    stats = {}

    for subdir, _, files in os.walk(base_dir):
        if os.path.basename(subdir).startswith(prefix):  # look only for runs against one enemy
            for f in files:
                if f == "stats.csv":
                    file_path = os.path.join(subdir, f)

                    try:
                       df = pd.read_csv(file_path)
                       df['original_index'] = range(len(df)) # to maintain the order

                       # separate records not from the islands phase
                       start_end_df = df[df['id'].isin(['start', 'end'])]

                        # filter out start and end
                       aggregated_df = df[~df['id'].isin(['start', 'end'])]
                       aggregated_df = aggregated_df.drop(columns=['id'], errors='ignore')

                       aggregated_df = aggregated_df.astype(float)
                       aggregated_df = aggregated_df.groupby('gen').mean().reset_index()

                       final_df = pd.concat([start_end_df, aggregated_df], ignore_index=True)

                       final_df = final_df.sort_values(by='original_index').drop(columns=['id'], errors='ignore')
                       final_df['gen'] = range(len(final_df))

                       stats[subdir] = final_df

                       # print(f"Last index value: {final_df['gen'].iloc[-1]}")

                    except Exception as e:
                        print(f, e)
    return stats

def plot_stats(stats_spec: dict, stats_gen: dict, enemy_group: list):
    save_path = "results2/plots/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    type = list(stats_spec.keys())[0]
    plot_name = ""
    if "1347" in type:
        plot_name = "1347"
    elif "2467" in type:
        plot_name = "2467"
    else:
        print("err")
        return

    # fit EA data
    all_best_fit_fit = []
    all_mean_fit_fit = []

    for _, df in stats_spec.items():
        all_best_fit_fit.append(df['best_fit'])
        all_mean_fit_fit.append(df['mean_fit'])


    best_fit_df_fit = pd.DataFrame(all_best_fit_fit)
    mean_fit_df_fit = pd.DataFrame(all_mean_fit_fit)

    avg_best_fit_fit = best_fit_df_fit.mean(axis=0)
    avg_mean_fit_fit = mean_fit_df_fit.mean(axis=0)

    std_best_fit_fit = best_fit_df_fit.std(axis=0)
    std_mean_fit_fit = mean_fit_df_fit.std(axis=0)
    generations = range(len(avg_best_fit_fit))

    # cheat EA data
    all_best_fit_cheat = []
    all_mean_fit_cheat = []

    for _, df in stats_gen.items():
        all_best_fit_cheat.append(df['best_fit'])
        all_mean_fit_cheat.append(df['mean_fit'])

    best_fit_df_cheat = pd.DataFrame(all_best_fit_cheat)
    mean_fit_df_cheat = pd.DataFrame(all_mean_fit_cheat)

    avg_best_fit_cheat = best_fit_df_cheat.mean(axis=0)
    avg_mean_fit_cheat = mean_fit_df_cheat.mean(axis=0)

    std_best_fit_cheat = best_fit_df_cheat.std(axis=0)
    std_mean_fit_cheat = mean_fit_df_cheat.std(axis=0)

    # plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('Generation', fontsize=16)
    ax1.set_ylabel('Fitness', fontsize=16)
    fig.suptitle(f'Average and best fitness for EAs trained on enemies {" ".join(str(x) for x in enemy_group)}', fontsize=18)

    # fit
    ax1.plot(generations, avg_best_fit_fit, label='best fitness spec', color='blue', lw=1)
    ax1.fill_between(generations, avg_best_fit_fit - std_best_fit_fit, avg_best_fit_fit + std_best_fit_fit,
                     color='blue', alpha=0.2)
    ax1.plot(generations, avg_mean_fit_fit, label='mean fitness spec', color='red', lw=1)
    ax1.fill_between(generations, avg_mean_fit_fit - std_mean_fit_fit, avg_mean_fit_fit + std_mean_fit_fit,
                     color='red', alpha=0.2)

    # cheat
    ax1.plot(generations, avg_best_fit_cheat, label='best fitness gen', color='blue', lw=1, linestyle='--')
    ax1.fill_between(generations, avg_best_fit_cheat - std_best_fit_cheat, avg_best_fit_cheat + std_best_fit_cheat,
                     color='blue', alpha=0.2)
    ax1.plot(generations, avg_mean_fit_cheat, label='mean fitness gen', color='red', lw=1, linestyle='--')
    ax1.fill_between(generations, avg_mean_fit_cheat - std_mean_fit_cheat, avg_mean_fit_cheat + std_mean_fit_cheat,
                     color='red', alpha=0.2)

    gens_start = GENS_TOGETHER_START
    gens_island = GENS_TOGETHER_START + GENS_ISLAND
    ax1.axvline(x=gens_start, color='brown', linestyle=':')
    ax1.axvline(x=gens_island, color='brown', linestyle=':')

    ax1.text(gens_start, ax1.get_ylim()[0], '          Island Phase', color='brown', fontsize=16,
             verticalalignment='bottom', horizontalalignment='left')
    ax1.text(gens_island, ax1.get_ylim()[0], '', color='brown', fontsize=12,
             verticalalignment='bottom', horizontalalignment='left')

    fig.legend(
        loc='lower right',
        bbox_to_anchor=(0.98, 0.34),
        fontsize=13,
        markerscale=2,
        # framealpha=1
        framealpha=0
    )

    ax1.grid(True)
    fig.tight_layout()

    plt.savefig(os.path.join(save_path, plot_name))
    print("saved " + plot_name)



RESULTS_DIR_NAME = "results2"
ENEMY_GROUPS = {
    "difficulty": [1, 3, 4, 7],
    "behavior": [2, 4, 6, 7],
}

if __name__ == "__main__":
    assert(os.path.exists(RESULTS_DIR_NAME))

    # plot fitness
    # behavior - 2467
    stats_spec2467 = read_stats_unprocessed(RESULTS_DIR_NAME, "spec-2467")
    stats_gen2467 = read_stats_unprocessed(RESULTS_DIR_NAME, "gen-2467")
    # print(list(stats_gen2467.values())[0].info()) #file names

    plot_stats(stats_spec2467, stats_gen2467, ENEMY_GROUPS["behavior"])

    # difficulty - 1347
    stats_spec1347 = read_stats_unprocessed(RESULTS_DIR_NAME, "spec-1347")
    stats_gen1347 = read_stats_unprocessed(RESULTS_DIR_NAME, "gen-1347")
    #
    plot_stats(stats_spec1347, stats_gen1347, ENEMY_GROUPS["difficulty"])