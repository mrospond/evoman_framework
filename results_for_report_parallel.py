import os

from params import *
import adaptative_ea_bad
import decreasing_ea
import concurrent.futures

results_dir = 'results'
runs = 10
min_mutation_str = f"{MIN_MUTATION % 1:.3f}".split('.')[1]
mutation_delta = f"{MUTATION_DELTA % 1:.3f}".split('.')[1]
doomsday_str = f"{DOOMSDAY % 1:.3f}".split('.')[1]

def adaptative(run):
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
        print("\n\nNEW GENERATION:", i)
        ea.run_generation()


def decreasing(run):
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
        print("\n\nNEW GENERATION:", i)
        ea.run_generation()

if __name__ == "__main__":

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # TODO: set max_workers, my cpu has 8 cores, so 5 processes at a time, to save some for the OS,
    #  and even if i set it to 8, 10 runs will still require 2 executions of this script (first 8 workers, then 2)
    max_workers = 5

    # adaptative()
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:  # Use ProcessPoolExecutor over ThreadPoolExecutor, cuz our task is CPU-bound
        executor.map(adaptative, range(runs))

    # decreasing()
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        executor.map(decreasing, range(runs))
