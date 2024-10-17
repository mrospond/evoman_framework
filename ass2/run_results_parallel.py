import multiprocessing
import concurrent.futures
import math

from complex_ea import ComplexEA
from global_params import *


enemy_groups = {
    "difficulty": [1, 3, 4, 7],
    "behavior": [2, 4, 6, 7],
}
names = {
    "fit": "spec",
    "cheat": "gen"
}

enemy_group = "difficulty"  # "difficulty" OR "behavior"
mode = "cheat"  # "fit" OR "cheat"


def run_complex(run):
    try:  # child process exceptions are not propagated to the parent process, so we need to catch and print them
        print(f"===== RUN {run} START=====")
        ea = ComplexEA(
            ea_name=names[mode],
            enemies=enemy_groups[enemy_group],
            mode=mode,
            run=run,
        )

        for _ in range(NUM_GENS_OVERALL):
            ea.run_generation()
        print(f"===== RUN {run} DONE =====")
    except Exception as e:
        print(f">>>>>>>>>>> [Error] in run {run}: {e}")

if __name__ == "__main__":
    # runs: https://docs.google.com/spreadsheets/d/1_QNvI1hB1JqzLJYi_Ikh9Y3Z5fvxNbkP_HiR3ZdixB0/edit?usp=sharing
    runs = 10

    # for r in range(runs):
    #     run_complex(r)

    n_cores_available = multiprocessing.cpu_count() - 2  # leave 2 cpus for OS
    # divide the tasks into equal chunks to optimize the load, based on available CPU cores.
    # e.g. 13 runs, 6 cores, min_rounds = 13/6 -> 3, max_workers = 13/3 -> 5
    min_rounds = math.ceil(runs / n_cores_available)
    max_workers = math.ceil(runs / min_rounds)
    max_workers = min(max_workers, n_cores_available)  # just in case but idk
    print(f"max_workers: {max_workers}")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Use ProcessPoolExecutor over ThreadPoolExecutor, cuz our task is CPU-bound
        executor.map(run_complex, range(runs))

    print("===== ALL DONE =====")
