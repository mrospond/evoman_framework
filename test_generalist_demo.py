import os
import matplotlib.pyplot as plt
import numpy as np
import adaptative_ea_generalist, adaptative_ea_bad

RESULTS_DIR_NAME="results/asg2"

def gain_4_all(weights_lines: list) -> dict:
    """
    returns game results for all enemies
    """
    enemies = [x for x in range(1,9)]

    weights = np.array(weights_lines, dtype=float)
    ea = adaptative_ea_generalist.SpecializedEA(RESULTS_DIR_NAME, enemies)
    fitness, playerlife, enemylife, time = ea.env.play(pcont=weights)
    print("fitness: ", fitness, "gain: ", playerlife - enemylife, "time: ", time)

    return (fitness, playerlife, enemylife, time)

def gain_4_single(weights_lines: list) -> float:
    """
    returns gain for a single enemy
    """
    enemies = [x for x in range(1,9)]

    experiment_name="test_experiment"
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    gain_dict = {}
    for enemy in enemies:
        weights = np.array(weights_lines, dtype=float)
        ea = adaptative_ea_bad.SpecializedEA(experiment_name, enemy) # single enemy version
        fitness, playerlife, enemylife, time = ea.env.play(pcont=weights)
        gain = playerlife - enemylife
        print("fitness: ", fitness, "gain: ", gain, "time: ", time)
        gain_dict[enemy] = gain

    return gain_dict

def plot_gain_against_each(gain_dict: dict):
    gains = list(gain_dict.values())
    labels = [f"{key}" for key in gain_dict.keys()]

    colors = ['coral' if i % 2 == 0 else 'crimson' for i in range(len(labels))]

    print(len(colors))

    fig, ax = plt.subplots()
    ax.set_ylabel('Individual Gain', fontsize=16)
    ax.set_xlabel('Enemy Number', fontsize=16)
    ax.set_title('Individual Gain per Enemy', fontsize=16)

    # Create the bar chart
    ax.bar(labels, gains, color=colors)

    plt.tight_layout()
    fig.autofmt_xdate()

    plt.savefig(os.path.join(RESULTS_DIR_NAME, "each_gain.png"))
    print("Saved plot as 'each_gain.png'")  


if __name__== "__main__":
    if not os.path.exists(RESULTS_DIR_NAME):
        os.makedirs(RESULTS_DIR_NAME)

    weights_file_path = "weights_1.txt"
    try:
        with open(weights_file_path, 'r') as file:
            weights_lines = file.readlines()
            weights_lines = [line.strip() for line in weights_lines]
    except Exception as e:
        print(e)
    # print(weights_lines)

    gain_4_all(weights_lines)
    plot_gain_against_each(gain_4_single(weights_lines))
