import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import adaptative_ea_generalist

def gain_4_all(weights: list) -> dict:
    """
    returns the gain for all enemies
    """
    enemies = [x for x in range(1,9)]

    experiment_name="test_experiment"
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    weights = np.array(weights_lines, dtype=float)
    ea = adaptative_ea_generalist.SpecializedEA(experiment_name, enemies)
    fitness, playerlife, enemylife, time = ea.env.play(pcont=weights)
    print("fitness: ", fitness, "gain: ", playerlife - enemylife, "time: ", time)

    return (fitness, playerlife, enemylife, time)


if __name__== "__main__":
    weights_file_path = "weights_1.txt"
    try:
        with open(weights_file_path, 'r') as file:
            weights_lines = file.readlines()
            weights_lines = [line.strip() for line in weights_lines]
    except Exception as e:
        print(e)
    # print(weights_lines)

    gain_4_all(weights_lines)
