################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

# imports framework
import sys, os, random
import numpy as np

from evoman.environment import Environment
from demo_controller import player_controller

DOM_UPPER = 1  # ??
DOM_LOWER = -1  # ??
POP_SIZE = 100
GENERATIONS = 30
MUTATION_THRESH = 0.2
DOOMSDAY = 0.25


class Demo():
    def __init__(self, experiment_name) -> None:
        n_hidden_neurons = 10

        self.env = Environment(
            experiment_name=experiment_name,
            playermode="ai",
            player_controller=player_controller(n_hidden_neurons),
            enemymode="static",
            level=2,
            speed="fastest",
            visuals=False,
        )

        self.n_weights = (self.env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

        self.gen_i = 0
        self.last_best = None
        self.last_best_fit = None
        self.mean_fit = None
        self.std_fit = None
        self.not_improved = 0

    def simulate(self, pop):
        fitness, playerlife, enemylife, time = self.env.play(pcont=pop)
        return fitness

    def get_fitness(self, pop):
        return np.array(list(map(lambda indiv: self.simulate(indiv), pop)))

    def gen_pop(self):
        pop = np.random.uniform(DOM_LOWER, DOM_UPPER, (POP_SIZE, self.n_weights))
        fit_pop = self.get_fitness(pop)
        return (pop, fit_pop)

    def tournament(self, pop, fit_pop, k):
        """"""
        selection = random.sample(range(0, POP_SIZE-1), k)

        best_fit = fit_pop[selection[0]]
        best_indiv = selection[0]
        for i in selection:
            if fit_pop[i] > best_fit:
                best_fit = fit_pop[i]
                best_indiv = pop[i]

        return best_indiv

    def limits(self, w):
        if w < DOM_LOWER:
            return DOM_LOWER
        if w > DOM_UPPER:
            return DOM_UPPER
        return w

    def norm(self, w, fit_pop):
        if ((max(fit_pop) - min(fit_pop))) > 0:
            w_norm = (w - min(fit_pop))/(max(fit_pop) - min(fit_pop))
        else:
            w_norm = 0

        if w_norm <= 0:
            w_norm = 0.0000000001

        return w_norm

    def reproduce(self, pop, fit_pop):
        all_offspring = np.zeros((0, self.n_weights))
        for _ in range(0, POP_SIZE, 2):  # pairs of parents
            p1 = self.tournament(pop, fit_pop, 2)
            p2 = self.tournament(pop, fit_pop, 2)

            n_offspring = np.random.randint(1, 3+1, 1)[0] # number of offspring
            offspring = np.zeros((n_offspring, self.n_weights)) # TODO WHYYYY

            for f in range(0, n_offspring):
                cross_prob = np.random.uniform(0, 1)
                offspring[f] = p1*cross_prob + p2*(1-cross_prob)

                # mutation
                for i in range(0, self.n_weights):
                    mutation_prob = np.random.uniform(0, 1)
                    if mutation_prob <= MUTATION_THRESH:
                        offspring[f][i] = offspring[f][i] + np.random.normal(0, 1)

                offspring[f] = np.array(list(map(lambda w: self.limits(w), offspring[f])))
                all_offspring = np.vstack((all_offspring, offspring[f]))

        return all_offspring

    def doomsday(self, pop, fit_pop):
        """Drastic restructuring: kill off DOOMSDAY*POP_SIZE individuals"""
        worst_n = np.round(DOOMSDAY*POP_SIZE)
        sorted_fit_pop = np.argsort(fit_pop)
        worst_fit_pop = sorted_fit_pop[0:worst_n]

        best_indiv = pop[sorted_fit_pop[-1]]

        for i in worst_fit_pop:
            for w in range(0, self.n_weights):
                prob_thresh = np.random.uniform(0, 1)
                if np.random.uniform(0, 1) <= prob_thresh:
                    pop[i][w] = np.random.uniform(DOM_LOWER, DOM_UPPER)
                else:
                    pop[i][w] = best_indiv[w]  # get value from best indiv

            fit_pop[i] = self.get_fitness([pop[i]])

        return (pop, fit_pop)

    def evolve_pop(self, pop, fit_pop):
        offspring = self.reproduce(pop, fit_pop)
        fit_offspring = self.get_fitness(offspring)

        pop = np.vstack((pop, offspring))
        fit_pop = np.append(fit_pop, fit_offspring)

        best_i = np.argmax(fit_pop)
        fit_pop[best_i] = float(self.get_fitness(np.array([pop[best_i]]))[0]) # repeats best eval, for stability issues
        best_fit = fit_pop[best_i]
        self.last_best = pop[best_i]

        # selection
        fit_pop_norm = np.array(list(map(lambda indiv: self.norm(indiv,fit_pop), fit_pop))) # avoiding negative probabilities, as fitness is ranges from negative numbers
        probs = (fit_pop_norm)/(fit_pop_norm.sum())
        chosen = np.random.choice(pop.shape[0], POP_SIZE-1, replace=False, p=probs)
        chosen = np.append(chosen[:-1], best_i)
        pop = pop[chosen]
        fit_pop = fit_pop[chosen]

        # searching new areas
        if self.last_best_fit is None or best_fit > self.last_best_fit:
            self.last_best_fit = best_fit
            self.not_improved = 0
        else:
            self.not_improved += 1

        if self.not_improved >= 15:
            # TODO: write doomsday
            print("DOOMSDAY")
            pop, fit_pop = self.doomsday(pop, fit_pop)
            self.not_improved = 0  # TODO technically didn't improve, another doomsday?

        return (pop, fit_pop)

    def stats(self, fit_pop):
        best_fit = np.max(fit_pop)
        mean_fit = np.mean(fit_pop)
        std_fit = np.std(fit_pop)

        print(f"{self.gen_i}:\tbest:{best_fit}\tmean_fit:{mean_fit}\tstd_fit:{std_fit}")

    def run_generation(self):
        if self.env.solutions is None:
            pop, fit_pop = self.gen_pop()
        else:
            pop, fit_pop = self.env.solutions

        new_pop, new_fit_pop = self.evolve_pop(pop, fit_pop)

        self.stats(new_fit_pop)
        # TODO bunch of writing to file stuff

        self.env.update_solutions([new_pop, new_fit_pop])
        self.env.save_state()

        self.gen_i += 1

    def show_best(self):
        self.env.update_parameter("visuals", True)
        self.env.update_parameter("speed", "normal")
        # self.get_fitness(np.array([self.last_best]))
        self.simulate(self.last_best)
        self.env.update_parameter("speed", "fastest")
        self.env.update_parameter("visuals", False)


if __name__ == "__main__":
    experiment_name = 'meret_test'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)


    demo = Demo(experiment_name)
    for i in range(GENERATIONS):
        demo.run_generation()
        if i % 10 == 0:
            demo.show_best()

    demo.show_best()
