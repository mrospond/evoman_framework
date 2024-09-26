import os, random, sys
import numpy as np
from numpy import ndarray
from scipy.stats import cauchy
from params import *

from evoman.environment import Environment
from demo_controller import player_controller

LIM_UPPER = 1
LIM_LOWER = -1
N_HIDDEN_NEURONS = 10


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
        self.mutation_prob = MAX_CROSSOVER - MIN_MUTATION

        self.last_best = None
        self.last_best_fit = None
        self.last_best_prob = None
        self.best_fit_since_dooms = None
        self.not_improved = 0  # TODO now this is best, could the mean

    def simulate(self, pop) -> float:
        fitness, playerlife, enemylife, time = self.env.play(pcont=pop)
        return fitness

    def get_fitness(self, pop) -> ndarray:
        return np.array([self.simulate(indiv) for indiv in pop])

    def gen_indiv(self, max_fit) -> ndarray:
        while True:
            indiv = np.random.uniform(LIM_LOWER, LIM_UPPER, (1, self.n_weights))
            fit = self.get_fitness(indiv)
            if fit <= max_fit:
                break

        return (indiv, fit)

    def gen_pop(self) -> tuple[ndarray, ndarray]:
        pop = np.random.uniform(LIM_LOWER, LIM_UPPER, (POP_SIZE, self.n_weights))
        fit_pop = self.get_fitness(pop)
        max_fit = 25
        for i in range(POP_SIZE):
            if fit_pop[i] > max_fit:
                indiv, fit = self.gen_indiv(max_fit)
                pop[i] = indiv
                fit_pop[i] = fit
        return (np.array(pop), fit_pop)

    def tournament(self, pop, fit_pop) -> ndarray:
        """
        Select best individual from random sample of "k" indivduals from pop.
        """
        sample = random.sample(range(0, POP_SIZE-1), TOURNAMENT_K)  # generate indexes of sampled individuals

        best_fit = fit_pop[sample[0]]
        best_i = sample[0]
        for i in sample:
            if fit_pop[i] > best_fit:
                best_fit = fit_pop[i]
                best_i = i

        return pop[best_i]

    def recombination(self, p1, p2) -> tuple[ndarray, ndarray]:
        """
        TODO explain
        """
        crossover_prob = MAX_CROSSOVER-self.mutation_prob  # probability of doing crossover with both parents

        if random.uniform(0, 1) <= crossover_prob:
            overlap_lower = int(random.uniform(0, self.n_weights - 2))
            overlap_higher = int(random.uniform(overlap_lower + 1, self.n_weights - 1))
            overlap = np.mean([p1[overlap_lower:overlap_higher], p2[overlap_lower:overlap_higher]], axis=0)
            c1 = np.append(p1[0:overlap_lower], np.append(overlap, p1[overlap_higher:]))
            c2 = np.append(p2[0:overlap_lower], np.append(overlap, p2[overlap_higher:]))
        else:
            c1 = np.copy(p1)
            c2 = np.copy(p2)

        return (c1, c2)

    def sample_cauchy(self, lower, upper) -> float:
        """
        Sample cauchy distribution with range [-2, 2] and samples with an absolute
        value of < MIN_MUTATION are forced to -MIN_MUTATION or MIN_MUTATION
        """
        r = cauchy.rvs()
        if np.abs(r < MIN_MUTATION):  # if |r| < MIN_MUTATION, it is set to MIN_MUTATION or -MIN_MUTATION
            if r < 0: r = -MIN_MUTATION
            if r >= 0: r = MIN_MUTATION
        elif r < lower or r > upper:  # r must be in range [lower, upper], if not, retry
            return self.sample_cauchy(lower, upper)

        return r

    def mutation(self, child) -> tuple[ndarray]:
        """
        Mutate
        """
        # Mutate weights
        for i in range(self.n_weights):
            if random.uniform(0, 1) <= self.mutation_prob:
                delta = self.sample_cauchy(-2, 2)
                new_val = LOWER_CAUCHY + (child[i]+delta + LOWER_CAUCHY) % 4  # wraparound in range [-2, 2]
                child[i] = min(LIM_UPPER, max(LIM_LOWER, new_val))

        return child

    def reproduce(self, pop, fit_pop) -> ndarray:
        all_offspring_weights = np.empty((0, self.n_weights))
        for _ in range(0, POP_SIZE, 2):  # go by pairs of parents
            p1 = self.tournament(pop, fit_pop)
            p2 = self.tournament(pop, fit_pop)

            c1, c2 = self.recombination(p1, p2)
            c1 = self.mutation(c1,)
            c2 = self.mutation(c2)

            all_offspring_weights = np.vstack((all_offspring_weights, c1))
            all_offspring_weights = np.vstack((all_offspring_weights, c2))

        return all_offspring_weights

    def reshuffle(self, pop, fit_pop) -> tuple[ndarray, ndarray]:
        """
        Drastic restructuring: kill off DOOMSDAY*POP_SIZE individuals and replace
        with randomly generated individuals.
        """
        print("\nDOOMSDAY!!!!!!!\n")

        worst_n = int(np.round(DOOMSDAY*POP_SIZE))
        sorted_fit_pop = np.argsort(fit_pop)
        worst_fit_pop_is = sorted_fit_pop[0:worst_n]

        for i in worst_fit_pop_is:
            for w in range(0, self.n_weights):
                pop[i][w] = np.random.uniform(LIM_LOWER, LIM_UPPER)

            fit_pop[i] = self.get_fitness([pop[i]])

        return (pop, fit_pop)

    def selection(self, pop, fit_pop) -> tuple[ndarray, ndarray]:
        """
        Exponential ranking-based selection.
        """
        sorted_pop_indices = np.argsort(fit_pop)
        pop_selection_probs = np.array(list(map(lambda i: 1 - np.e**(-i), sorted_pop_indices)))
        pop_selection_probs /= np.sum(pop_selection_probs)  # make sure all probs sum to 1
        chosen = np.random.choice(pop.shape[0], POP_SIZE, p=pop_selection_probs, replace=False)

        pop = pop[chosen]
        fit_pop = fit_pop[chosen]

        return (pop, fit_pop)

    def get_stable_best(self, pop, fit_pop) -> int:
        """
        Returns index of individual in 'pop' with the best fitness value that
        is also stable across two runs.
        """
        best_i = np.argmax(fit_pop)
        val1 = fit_pop[best_i]
        val2 = self.get_fitness([pop[best_i]])[0]
        if abs(val2 - val1) > MAX_DIFF_STABLE:
            print("rejecting", val1)
            fit_pop[best_i] = np.mean([val1, val2])
            return self.get_stable_best(pop, fit_pop)

        return best_i

    def evolve_pop(self, pop, fit_pop) -> tuple[ndarray, ndarray]:
        offspring = self.reproduce(pop, fit_pop)
        fit_offspring = self.get_fitness(offspring)

        # Select from both parents and offspring
        alltogether = np.vstack((pop, offspring))
        fit_alltogether = np.append(fit_pop, fit_offspring)
        new_pop, new_pop_fit = self.selection(alltogether, fit_alltogether)

        if self.best_fit_since_dooms is None or max(new_pop_fit) > self.best_fit_since_dooms:
            self.best_fit_since_dooms = max(new_pop_fit)
            self.not_improved = 0
        else:
            self.not_improved += 1

        print("not improved", self.not_improved)
        if self.not_improved >= DOOMSDAY_GENS:
            new_pop, new_pop_fit = self.reshuffle(new_pop, new_pop_fit)
            self.best_fit_since_dooms = None

        new_best_i = self.get_stable_best(new_pop, new_pop_fit)

        self.last_best = new_pop[new_best_i]
        self.last_best_fit = new_pop_fit[new_best_i]

        return (new_pop, new_pop_fit)

    def stats(self, fit_pop):
        """
        TODO very basic, could be better :)
        """
        best_fit = np.max(fit_pop)  # self.last_best_fit
        mean_fit = np.mean(fit_pop)
        std_fit = np.std(fit_pop)

        if self.best_fit_since_dooms is None and self.gen > 0:
            doomsday = 1
        else:
            doomsday = 0

        print(f"{self.gen}:\tbest_fit:{best_fit}\tmean_fit:{mean_fit}\tstd_fit:{std_fit}\tprob:{self.mutation_prob}")
        with open(f"{self.env.experiment_name}/stats.csv", "a+") as f:
            f.write(f"{self.gen},{best_fit},{mean_fit},{std_fit},{self.mutation_prob},{doomsday}\n")

        with open(f"{self.env.experiment_name}/weights.csv", "a+") as f:
            f.write(",".join([str(w) for w in self.last_best]) + "\n")

    def run_generation(self):
        if self.env.solutions is None:
            new_pop, new_fit_pop = self.gen_pop()
            new_best_i = self.get_stable_best(new_pop, new_fit_pop)
            self.last_best = new_pop[new_best_i]
            self.last_best_fit = new_fit_pop[new_best_i]
            self.best_fit_since_dooms = self.last_best_fit
        else:
            pop, fit_pop = self.env.solutions
            new_pop, new_fit_pop = self.evolve_pop(pop, fit_pop)

        self.stats(new_fit_pop)

        self.env.update_solutions([new_pop, new_fit_pop])
        self.env.save_state()

        self.gen += 1
        self.mutation_prob = min(MAX_CROSSOVER-MIN_MUTATION, max(MIN_MUTATION, self.mutation_prob - MUTATION_DELTA))

    def show_best(self):
        self.env.update_parameter("visuals", True)
        self.env.update_parameter("speed", "normal")
        self.get_fitness(np.array([self.last_best]))
        self.env.update_parameter("speed", "fastest")
        self.env.update_parameter("visuals", False)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        ENEMY = sys.argv[1]

    min_mutation_str = f"{MIN_MUTATION % 1:.3f}".split('.')[1]
    mutation_delta = f"{MUTATION_DELTA % 1:.3f}".split('.')[1]
    doomsday_str = f"{DOOMSDAY % 1:.3f}".split('.')[1]

    i = 0
    while True:
        experiment_name = f"decreasing_ea_{i}-{ENEMY}-{POP_SIZE}-{DOOMSDAY_GENS}-{doomsday_str}-{min_mutation_str}-{TOURNAMENT_K}-{LOWER_CAUCHY}-{UPPER_CAUCHY}-{mutation_delta}"
        if not os.path.exists(experiment_name):
            break
        else:
            i += 1

    experiment_name = f"decreasing_ea_{i}-{ENEMY}-{POP_SIZE}-{DOOMSDAY_GENS}-{doomsday_str}-{min_mutation_str}-{TOURNAMENT_K}-{LOWER_CAUCHY}-{UPPER_CAUCHY}-{mutation_delta}"
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    with open(f"{experiment_name}/stats.csv", "w+") as f:
        f.write("gen,best_fit,mean_fit,std_fit,prob,doomsday\n")

    with open(f"{experiment_name}/weights.csv", "w+") as f:
        f.write("")

    ea = SpecializedEA(experiment_name, ENEMY)
    for i in range(GENERATIONS):
        print("\n\nNEW GENERATION")
        ea.run_generation()
        if i % 10 == 0:
            ea.show_best()

    ea.show_best()
