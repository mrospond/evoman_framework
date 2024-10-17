import json
import numpy as np
import os
import random

import global_params

from global_params import *
from numpy import ndarray
from single_ea import SingleEA

headless = True #Change to false if you want to visualize at some points. Not sure if it can be switched on and off on the fly
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


class ComplexEA():
    def __init__(self, ea_name: str, enemies: list[int], mode: str) -> None:
        self.enemies = enemies
        self.mode = mode.strip().lower()
        assert self.mode in ["random", "fit", "cheat"]

        # Create folder
        enemies_str = "".join([str(e) for e in enemies])
        i = 0
        while True:
            experiment_name = f"./results/{ea_name}-{enemies_str}_{i}"
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

        with open(f"{experiment_name}/weights_competition.csv", "w+") as f:
            f.write("")

        with open(f"{experiment_name}/weights.csv", "w+") as f:
            f.write("")

        print("done making folders")

        self.gen = 0
        self.EAs: list[SingleEA] = None
        self.generalist_EA = SingleEA(
            id="generalist",
            experiment_name=self.experiment_name,
            enemies=[1, 2, 3, 4, 5, 6, 7, 8],
            pop_size=0,
            pop=np.array([]),
        )

    def get_enemy_probs(self):
        """
        Returns a dictionary with the average inverse fitness (100-fit) of the last generation
        of the current EA for each enemy in self.enemies, normalized so they all add up to 1.
        """
        pop = self.EAs[0].pop

        inv_fit_dict = {}
        for e in self.enemies:
            ea = SingleEA(
                id="temp",
                experiment_name=self.experiment_name,
                enemies=[e],
                pop_size=len(pop),
                pop=pop,
                communicates=False,
            )
            avg_fit = np.average(ea.get_fitness(pop))
            inv_fit_dict[e] = max(5, 100 - avg_fit)  # Try to avoid almost-zero probabilities

        norm_dict = {}
        for key, val in inv_fit_dict.items():
            norm_dict[key] = val/np.sum(list(inv_fit_dict.values()))

        with open(f"{self.experiment_name}/enemy_probs.json", "w+") as f:
            json.dump(norm_dict, f)

        return norm_dict

    def save_best_generalist(self, pop: ndarray) -> None:
        """
        Returns individuals from pop that performs best for all 8 enemies.
        """
        pop_fit = self.generalist_EA.get_fitness(pop)

        best_i = np.argmax(pop_fit)
        best_weights = pop[best_i]
        best_fitness = pop_fit[best_i]

        with open(f"{self.experiment_name}/weights_competition.csv", "a") as f:
            f.write(str(best_fitness) + "," + ",".join([str(w) for w in best_weights]) + "\n")

    def run_generation(self) -> None:
        if self.gen == 0:
            start_EA = SingleEA(
                "start", self.experiment_name, self.enemies, POP_SIZE
            )
            self.EAs = [start_EA]
        elif self.gen == (GENS_TOGETHER_START + GENS_ISLAND):  # last phase (together)
            global TOURNAMENT_K
            TOURNAMENT_K = END_TOURNAMENT_K

            pop = np.empty((0, self.EAs[0].n_weights))
            for ea in self.EAs:
                pop = np.vstack((pop, ea.pop))
            if len(pop) > POP_SIZE:
                pop_is = random.sample(range(len(pop)), POP_SIZE)
                selected_pop = np.zeros((0, self.EAs[0].n_weights))
                for i in pop_is:
                    selected_pop = np.vstack((selected_pop, pop[i]))

                pop = selected_pop

            end_EA = SingleEA(
                id="end",
                experiment_name=self.experiment_name,
                enemies=self.enemies,
                pop_size=POP_SIZE,
                pop=pop
            )
            self.EAs = [end_EA]
        elif self.gen == GENS_TOGETHER_START:
            island_enemies = []
            if self.mode == "random":
                for _ in range(NUM_ISLANDS):
                    island_enemies.append(random.sample(self.enemies, NUM_ENEMIES))
            elif self.mode == "fit" or self.mode == "cheat":
                enemy_probs = self.get_enemy_probs()
                for _ in range(NUM_ISLANDS):
                    island_enemies.append(np.random.choice(list(enemy_probs.keys()), NUM_ENEMIES, p=list(enemy_probs.values()), replace=False))
            elif self.mode == "cheat":
                enemy_probs = self.get_enemy_probs()
                for _ in range(NUM_ISLANDS - 1):
                    island_enemies.append(np.random.choice(list(enemy_probs.keys()), NUM_ENEMIES, p=list(enemy_probs.values()), replace=False))

            new_EAs = []
            island_size = int(POP_SIZE / NUM_ISLANDS)
            if self.mode == "cheat":
                start_i = 1
                cheat_ea = SingleEA(
                    id="gi",
                    experiment_name=self.experiment_name,
                    enemies=self.enemies,
                    pop_size=island_size,
                    pop=self.EAs[0].pop[:island_size],
                    communicates=True,
                )
                new_EAs.append(cheat_ea)
            else:
                start_i = 0

            for i in range(start_i, NUM_ISLANDS):
                ea_id = f"si{i}"
                pop = self.EAs[0].pop[i*island_size:(i+1)*island_size]
                ea = SingleEA(
                    id=ea_id,
                    experiment_name=self.experiment_name,
                    enemies=island_enemies[i],
                    pop_size=island_size,
                    pop=pop,
                    communicates=True
                )
                new_EAs.append(ea)

            self.EAs = new_EAs

        if len(self.EAs) == 1:
            self.EAs[0].run_generation()
        else:
            for ea in self.EAs:
                ea.run_generation()

        pop_all = np.zeros((0, self.EAs[0].n_weights))
        for ea in self.EAs:
            pop_all = np.vstack((pop_all, ea.pop))
            if ea.communicates and ea.gen > 0 and ea.gen % EPOCH == 0:  # migration
                print("MIGRATION")
                for ea in self.EAs:
                    migrants_rank_based, migrants_random = ea.emigration()
                    # Copy migrants into other EAs
                    for immi_ea in self.EAs:
                        if ea.id == immi_ea.id:
                            continue
                        if ea.id == "gi":  # generalized island receives rank-based migrants
                            immi_ea.immigration(migrants_rank_based)
                        else:  # specialized islands receive random migrants
                            immi_ea.immigration(migrants_random)

        if self.gen % SAVE_GENERALIST == 0:
            self.save_best_generalist(pop_all)

        self.gen += 1


if __name__ == "__main__":
    SAVE_GENERALIST = 1
    complex_ea = ComplexEA("generalized", [1, 3, 4, 7], "cheat")
    for i in range(NUM_GENS_OVERALL):
        complex_ea.run_generation()
