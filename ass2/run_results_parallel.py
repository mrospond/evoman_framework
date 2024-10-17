import multiprocessing
import concurrent.futures

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
    print(f"===== RUN {run} =====")

    ea = ComplexEA(
        ea_name=names[mode],
        enemies=enemy_groups[enemy_group],
        mode=mode,
    )

    for _ in range(NUM_GENS_OVERALL):
        ea.run_generation()


if __name__ == "__main__":
    runs = 2

    # for r in range(runs):
    #     run_complex(r)

    max_workers = multiprocessing.cpu_count() - 2  # leave 2 cpus for OS

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Use ProcessPoolExecutor over ThreadPoolExecutor, cuz our task is CPU-bound
        executor.map(run_complex, range(runs))

    print("===== ALL DONE =====")
