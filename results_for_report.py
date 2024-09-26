import os
from params import *
import adaptative_ea_bad
import decreasing_ea

results_dir = 'results'
runs = 10
min_mutation_str = f"{MIN_MUTATION % 1:.3f}".split('.')[1]
mutation_delta = f"{MUTATION_DELTA % 1:.3f}".split('.')[1]
doomsday_str = f"{DOOMSDAY % 1:.3f}".split('.')[1]

def adaptative():
    for run in range(runs):

        print(f"===== RUN {run} =====")
        experiment_name = os.path.join(results_dir,
            f"adaptative_ea_bad/adaptative_ea_bad-{ENEMY}-{POP_SIZE}-{DOOMSDAY_GENS}-{doomsday_str}-{min_mutation_str}-{TOURNAMENT_K}-{LOWER_CAUCHY}-{UPPER_CAUCHY}-{mutation_delta}-{run}")
        if not os.path.exists(experiment_name):
            os.makedirs(experiment_name)

        with open(f"{experiment_name}/stats.csv", "w+") as f:
            f.write("gen,best_fit,mean_fit,std_fit,best_prob,mean_prob,median_prob,std_prob,doomsday\n")

        with open(f"{experiment_name}/weights.csv", "w+") as f:
            f.write("")

        ea = adaptative_ea_bad.SpecializedEA(experiment_name, ENEMY)
        for i in range(GENERATIONS):
            print("\n\nNEW GENERATION")
            ea.run_generation()

def decreasing():
    for run in range(runs):
        print(f"===== RUN {run} =====")
        experiment_name = os.path.join(results_dir,
            f"decreasing_ea/decreasing_ea-{ENEMY}-{POP_SIZE}-{DOOMSDAY_GENS}-{doomsday_str}-{min_mutation_str}-{TOURNAMENT_K}-{LOWER_CAUCHY}-{UPPER_CAUCHY}-{mutation_delta}-{run}")
        if not os.path.exists(experiment_name):
            os.makedirs(experiment_name)

        with open(f"{experiment_name}/stats.csv", "w+") as f:
            f.write("gen,best_fit,mean_fit,std_fit,prob,doomsday\n")

        with open(f"{experiment_name}/weights.csv", "w+") as f:
            f.write("")

        ea = decreasing_ea.SpecializedEA(experiment_name, ENEMY)
        for i in range(GENERATIONS):
            print("\n\nNEW GENERATION")
            ea.run_generation()


if __name__ == "__main__":

    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # adaptative()
    decreasing()