import sys, os, random
import numpy as np
from numpy import ndarray

from evoman.environment import Environment
from demo_controller import player_controller

LIM_UPPER = 1
LIM_LOWER = -1
N_HIDDEN_NEURONS = 10
POP_SIZE = 100
GENERATIONS = 200
MUTATION_PROB = 0.3
DOOMSDAY = 0.4  # PART OF POPULATION THAT GETS DESTROYED DURING RESHUFFLE

class SpecializedEA():
    def __init__(self, experiment_name, enemy) -> None:
        self.env = Environment(
            experiment_name=experiment_name,
            enemies=[enemy],
            level=2,  # do not change
            playermode="ai",  # do not change
            enemymode="static",  # do not change
            contacthurt="player",  # do not change
            player_controller=player_controller(N_HIDDEN_NEURONS),
            speed="fastest",
            visuals=False,
        )

        self.n_weights = (self.env.get_num_sensors()+1)*N_HIDDEN_NEURONS + (N_HIDDEN_NEURONS+1)*5
        self.gen = 0
        self.last_best = None
        self.last_best_fit = None
        self.not_improved = 0

    def limits(self, w) -> float:  # TODO literally their code
        if w < LIM_LOWER: return LIM_LOWER
        if w > LIM_UPPER: return LIM_UPPER
        return w

    def norm(self, w, fit_pop) -> float:  # TODO literally their code
        if ((max(fit_pop) - min(fit_pop))) > 0:
            w_norm = (w - min(fit_pop))/(max(fit_pop) - min(fit_pop))
        else:
            w_norm = 0

        if w_norm <= 0:
            w_norm = 0.0000000001

        return w_norm

    def simulate(self, pop) -> float:
        fitness, playerlife, enemylife, time = self.env.play(pcont=pop)
        return fitness

    def get_fitness(self, pop) -> ndarray:
        return np.array([self.simulate(indiv) for indiv in pop])

    def gen_pop(self) -> tuple[ndarray, ndarray]:
        pop = np.random.uniform(LIM_LOWER, LIM_UPPER, (POP_SIZE, self.n_weights))
        fit_pop = self.get_fitness(pop)

        return (pop, fit_pop)

    def tournament(self, pop, fit_pop, k) -> ndarray:
        """
        Select best individual from random sample of "k" indivduals from pop.
        """
        sample = random.sample(range(0, POP_SIZE-1), k)  # generate indexes of sampled individuals

        best_fit = fit_pop[sample[0]]
        best_indiv = sample[0]
        for i in sample:
            if fit_pop[i] > best_fit:
                best_fit = fit_pop[i]
                best_indiv = i

        return pop[best_indiv]

    def recombination(self, p1, p2) -> ndarray:
        """
        TODO crossover
        """
        return p1*0.5+p2*0.5

    def mutation(self, indiv) -> ndarray:
        """
        TODO mutation
        """
        for i in range(self.n_weights):
            if random.uniform(0, 1) <= MUTATION_PROB:
                indiv[i] = random.uniform(LIM_UPPER, LIM_LOWER)

        return indiv

    def reproduce(self, pop, fit_pop, tournament_k, min_offspring, max_offspring):
        all_offspring = np.empty((0, self.n_weights))
        for _ in range(0, POP_SIZE, 2):  # go by pairs of parents
            p1 = self.tournament(pop, fit_pop, tournament_k)
            p2 = self.tournament(pop, fit_pop, tournament_k)

            n_children = random.sample(range(min_offspring, max_offspring+1), 1)[0]  # TODO: do we want dynamic number of offspring?
            for o in range(0, n_children):
                new_child = self.recombination(p1, p2)
                new_child = self.mutation(new_child)
                new_child = [self.limits(w) for w in new_child]

                all_offspring = np.vstack((all_offspring, new_child))

        return all_offspring

    def reshuffle(self, pop, fit_pop) -> tuple[ndarray, ndarray]:
        """
        Drastic restructuring: kill off DOOMSDAY*POP_SIZE individuals.
        TODO: current is their implementation
        """
        worst_n = np.round(DOOMSDAY*POP_SIZE)
        sorted_fit_pop = np.argsort(fit_pop)
        worst_fit_pop = sorted_fit_pop[0:worst_n]

        best_indiv = pop[sorted_fit_pop[-1]]

        for i in worst_fit_pop:
            for w in range(0, self.n_weights):
                prob_thresh = np.random.uniform(0, 1)  # TODO: maybe make prob threshold relative to generation number (higher prob of random at start)
                if np.random.uniform(0, 1) <= prob_thresh:
                    pop[i][w] = np.random.uniform(LIM_LOWER, LIM_UPPER)
                else:
                    pop[i][w] = best_indiv[w]  # get value from best indiv

            fit_pop[i] = self.get_fitness([pop[i]])

        return (pop, fit_pop)

    def selection(self, offspring, fit_offspring, best_i) -> tuple[ndarray, ndarray]:
        """
        TODO selection, now just get best from offspring
        """
        sorted_fit_offspring = np.argsort(fit_offspring)
        selected_fit_offspring = sorted_fit_offspring[-POP_SIZE-1:-1]
        if best_i not in selected_fit_offspring:
            selected_fit_offspring[-1] = best_i

        pop = np.array([offspring[i] for i in selected_fit_offspring])
        pop_fit = np.array([fit_offspring[i] for i in selected_fit_offspring])

        return (pop, pop_fit)

    def evolve_pop(self, pop, fit_pop, reshuffle_t) -> tuple[ndarray, ndarray]:
        offspring = self.reproduce(pop, fit_pop, 10, 2, 5)
        fit_offspring = self.get_fitness(offspring)

        best_i = np.argmax(fit_offspring)
        fit_offspring[best_i] = float(self.get_fitness(np.array([offspring[best_i]]))[0])  # repeats best eval, for stability issues TODO literally their code
        best_fit = fit_offspring[best_i]
        self.last_best = offspring[best_i]

        new_pop, new_pop_fit = self.selection(offspring, fit_offspring, best_i)

        if self.last_best_fit is None or best_fit > self.last_best_fit:
            self.last_best_fit = best_fit
            self.not_improved = 0
        else:
            self.not_improved += 1

        if self.not_improved >= reshuffle_t:
            new_pop, new_pop_fit = self.reshuffle(new_pop, new_pop_fit)

        return (new_pop, new_pop_fit)

    def stats(self, fit_pop):
        """
        TODO very basic, could be better :)
        """
        best_fit = np.max(fit_pop)
        mean_fit = np.mean(fit_pop)
        std_fit = np.std(fit_pop)

        print(f"{self.gen}:\tbest:{best_fit}\tmean_fit:{mean_fit}\tstd_fit:{std_fit}")

    def run_generation(self):
        if self.env.solutions is None:
            pop, fit_pop = self.gen_pop()
        else:
            pop, fit_pop = self.env.solutions

        new_pop, new_fit_pop = self.evolve_pop(pop, fit_pop, 10)

        self.stats(new_fit_pop)
        # TODO bunch of writing to file stuff

        self.env.update_solutions([new_pop, new_fit_pop])
        self.env.save_state()

        self.gen += 1

    def show_best(self):
        self.env.update_parameter("visuals", True)
        self.env.update_parameter("speed", "normal")
        # self.get_fitness(np.array([self.last_best]))
        self.simulate(self.last_best)
        self.env.update_parameter("speed", "fastest")
        self.env.update_parameter("visuals", False)

if __name__ == "__main__":
    experiment_name = 'specialized_ea'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    demo = SpecializedEA(experiment_name, 1)
    for i in range(GENERATIONS):
        demo.run_generation()
        if i % 10 == 0:
            demo.show_best()

    demo.show_best()
