import os
import matplotlib.pyplot as plt
import numpy as np
import adaptative_ea_asg2

from evoman.environment import Environment
from demo_controller import player_controller


RESULTS_DIR_NAME="results/asg2"
N_HIDDEN_NEURONS=10

def gain_4_all(env: Environment, weights_lines: list) -> tuple:
    """
    returns game results against all enemies
    """
    enemies = [x for x in range(1,9)]

    weights = np.array(weights_lines, dtype=float)
    ea = adaptative_ea_asg2.SpecializedEA(env, RESULTS_DIR_NAME, enemies=enemies)
    fitness, playerlife, enemylife, time = ea.env.play(pcont=weights)
    print("fitness: ", fitness, "gain: ", playerlife - enemylife, "time: ", time)

    return (fitness, playerlife, enemylife, time)

def gain_4_single(env: Environment, weights_lines: list) -> dict:
    """
    returns gain for each enemy
    """
    enemies = [x for x in range(1,9)]

    gain_dict = {}
    for enemy in enemies:
        weights = np.array(weights_lines, dtype=float)
        ea = adaptative_ea_asg2.SpecializedEA(env, RESULTS_DIR_NAME, enemies=[enemy]) # single enemy version
        fitness, playerlife, enemylife, time = ea.env.play(pcont=weights)
        gain = playerlife - enemylife
        print("fitness: ", fitness, "gain: ", gain, "time: ", time)
        gain_dict[enemy] = gain

    return gain_dict

def plot_gain_against_each(gain_dict: dict):
    gains = list(gain_dict.values())
    labels = [f"{key}" for key in gain_dict.keys()]

    colors = ['coral' if i % 2 == 0 else 'crimson' for i in range(len(labels))]

    fig, ax = plt.subplots()
    ax.set_ylabel('Individual Gain', fontsize=16)
    ax.set_xlabel('Enemy Number', fontsize=16)
    ax.set_title('Individual Gain per Enemy', fontsize=16)

    # Create the bar chart
    ax.bar(labels, gains, color=colors)

    plt.tight_layout()
    # fig.autofmt_xdate()

    plt.savefig(os.path.join(RESULTS_DIR_NAME, "gain_each.png"))
    print("Saved plot as 'gain_each.png'")  


if __name__== "__main__":
    if not os.path.exists(RESULTS_DIR_NAME):
        os.makedirs(RESULTS_DIR_NAME)

    env_single = Environment(
            experiment_name=RESULTS_DIR_NAME,
            level=2,  # do not change
            playermode="ai",  # do not change
            enemymode="static",  # do not change
            contacthurt="player",  # do not change
            player_controller=player_controller(N_HIDDEN_NEURONS),
            speed="fastest",
            visuals=False,
        )
    
    env_multi = Environment(
            experiment_name=RESULTS_DIR_NAME,
            multiplemode="yes",
            playermode="ai",
            player_controller=player_controller(N_HIDDEN_NEURONS),
            enemymode="static",
            level=2,
            speed="fastest",
            visuals=False,
            contacthurt="player" # do not change
        )

    weights_file_path = "weights_1.txt"
    try:
        with open(weights_file_path, 'r') as file:
            weights_lines = file.readlines()
            weights_lines = [line.strip() for line in weights_lines]
    except Exception as e:
        print(e)
    # print(weights_lines)

    gain_4_all(env_multi, weights_lines)
    plot_gain_against_each(gain_4_single(env_single, weights_lines))
