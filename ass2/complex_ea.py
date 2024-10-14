import json
import numpy as np
import os
import random

import global_params

from global_params import *
from single_ea import SingleEA


class ComplexEA():
    def __init__(self, ea_name: str, enemies: list[int]) -> None:
        self.enemies = enemies

        # Create folder
        enemies_str = "".join([str(e) for e in enemies])
        i = 0
        while True:
            experiment_name = f"{ea_name}_{i}-{enemies_str}"
            if not os.path.exists(experiment_name):
                break
            else:
                i += 1
            pass

        self.experiment_name = experiment_name
        os.makedirs(experiment_name)
        with open(f"{experiment_name}/config.json", "w+") as f:
            params = {key: value for key, value in vars(global_params).items()
                      if not (key.startswith("__"))}
            json.dump(params, f)

        with open(f"{experiment_name}/stats.csv", "w+") as f:
            f.write("gen,id,enemies,best_fit,mean_fit,std_fit\n")

        with open(f"{experiment_name}/weights.csv", "w+") as f:
            f.write("")

        # TODO: think of a title
        self.gen = 0
        self.EAs: list[SingleEA] = None

    def run_generation(self) -> None:
        if self.gen == 0:
            start_EA = SingleEA(
                0, self.experiment_name, self.enemies, POP_SIZE
            )
            self.EAs = [start_EA]
        elif NUM_ISLANDS > 1 and self.gen == (GENS_TOGETHER_START + GENS_ISLAND):
            pop = np.empty((0, self.EAs[0].n_weights))  # TODO
            for ea in self.EAs:
                pop = np.vstack((pop, ea.pop))
            if len(pop) > POP_SIZE:  # TODO: do something if len(pop) > pop_size
                pop_is = random.sample(range(len(pop)), POP_SIZE)
                selected_pop = np.zeros((0, self.EAs[0].n_weights))
                for i in pop_is:
                    pop_is = np.vstack((selected_pop, pop[i]))

                pop = selected_pop

            end_EA = SingleEA(
                2, self.experiment_name, self.enemies, POP_SIZE, pop
            )
            self.EAs = [end_EA]
        elif NUM_ISLANDS > 1 and self.gen == GENS_TOGETHER_START:
            new_EAs = []
            for i in range(NUM_ISLANDS):
                ea_id = 10 + i
                pop_size = int(POP_SIZE / NUM_ISLANDS)
                pop = self.EAs[0].pop[i*pop_size:(i+1)*pop_size]
                ea_enemies = random.sample(self.enemies, NUM_ENEMIES)
                ea = SingleEA(
                    id=ea_id,
                    experiment_name=self.experiment_name,
                    enemies=ea_enemies,
                    pop_size=pop_size,
                    pop=pop,
                    communicates=True
                )
                new_EAs.append(ea)

            self.EAs = new_EAs

        for ea in self.EAs:
            ea.run_generation()
            # TODO: get best individual?
            if ea.communicates and ea.gen % EPOCH == 0:  # migration
                # TODO: migration
                pass

        self.gen += 1

if __name__ == "__main__":
    complex_ea = ComplexEA("firsttry", ENEMIES)
    for i in range(NUM_GENS_OVERALL):
        complex_ea.run_generation()
