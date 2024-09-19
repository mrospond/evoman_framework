import os, random
import numpy as np
from numpy import ndarray
from scipy.stats import cauchy

from evoman.environment import Environment
from demo_controller import player_controller

LIM_UPPER = 1
LIM_LOWER = -1
N_HIDDEN_NEURONS = 10
POP_SIZE = 100
GENERATIONS = 200

# TWEAK:
MIN_MUTATION = 0.001  # minimum value of mutation
TOURNAMENT_K = 10
LOWER_CAUCHY = -2
UPPER_CAUCHY = 2
# DOOMSDAY = 0.4  # PART OF POPULATION THAT GETS DESTROYED DURING RESHUFFLE


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
        self.last_best_prob = None
        self.not_improved = 0  # TODO now this is best, could the mean

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

    def gen_pop(self) -> tuple[ndarray, ndarray, ndarray]:
        pop = np.random.uniform(LIM_LOWER, LIM_UPPER, (POP_SIZE, self.n_weights))
        prob_pop = np.random.uniform(0, 1, POP_SIZE)
        fit_pop = self.get_fitness(pop)

        return (pop, fit_pop, prob_pop)

    def tournament(self, pop, fit_pop, prob_pop) -> tuple[ndarray, ndarray, ndarray]:
        """
        Select best individual from random sample of "k" indivduals from pop.
        """
        sample = random.sample(range(0, POP_SIZE-1), TOURNAMENT_K)  # generate indexes of sampled individuals

        best_fit = fit_pop[sample[0]]
        best_indiv = sample[0]
        for i in sample:
            if fit_pop[i] > best_fit:
                best_fit = fit_pop[i]
                best_indiv = i

        return (pop[best_indiv], fit_pop[best_indiv], prob_pop[best_indiv])

    def recombination(self, mutation_prob, p1, p1_prob, p2, p2_prob) -> tuple[ndarray, ndarray, ndarray, ndarray]:
        """
        TODO explain
        """
        crossover_prob = 1-mutation_prob  # probability of doing crossover with both parents

        if random.uniform(0, 1) <= crossover_prob:
            overlap_lower = int(random.uniform(0, self.n_weights - 2))
            overlap_higher = int(random.uniform(overlap_lower + 1, self.n_weights - 1))
            overlap = np.mean([p1[overlap_lower:overlap_higher], p2[overlap_lower:overlap_higher]], axis=0)
            c1 = np.append(p1[0:overlap_lower], np.append(overlap, p1[overlap_higher:]))
            c2 = np.append(p2[0:overlap_lower], np.append(overlap, p2[overlap_higher:]))
            c1_prob = mutation_prob
            c2_prob = mutation_prob
        else:
            c1 = p1
            c2 = p2
            c1_prob = p1_prob
            c2_prob = p2_prob

        return (c1, c1_prob, c2, c2_prob)

    def sample_cauchy(self) -> float:
        """
        Sample cauchy distribution with range [-2, 2] and samples with an absolute
        value of < MIN_MUTATION are forced to -MIN_MUTATION or MIN_MUTATION
        """
        r = cauchy.rvs()
        if np.abs(r < MIN_MUTATION):  # if |r| < MIN_MUTATION, it is set to MIN_MUTATION or -MIN_MUTATION
            if r < 0: r = -MIN_MUTATION
            if r >= 0: r = MIN_MUTATION
        elif r < LOWER_CAUCHY or r > UPPER_CAUCHY:  # r must be in range [-2, 2]
            return self.sample_cauchy()

        return r

    def mutation(self, mutation_prob, child, c_prob) -> tuple[ndarray, float]:
        """
        TODO mutation
        """
        # Mutate weights
        for i in range(self.n_weights):
            if random.uniform(0, 1) <= mutation_prob:
                delta = self.sample_cauchy()
                child[i] = LOWER_CAUCHY + (child[i]+delta + LOWER_CAUCHY) % 4  # wraparound in range [-2, 2]

        # Mutate mutation probability
        if random.uniform(0, 1) <= mutation_prob:
            delta = self.sample_cauchy()
            mutation_prob = (mutation_prob+delta)%1  # wraparound in range [0, 1]

        return child, mutation_prob

    def reproduce(self, pop, fit_pop, prob_pop) -> tuple[ndarray, ndarray]:
        all_offspring_weights = np.empty((0, self.n_weights))
        all_offspring_mutation_probs = []
        for _ in range(0, POP_SIZE, 2):  # go by pairs of parents
            p1, p1_fit, p1_prob = self.tournament(pop, fit_pop, prob_pop)
            p2, p2_fit, p2_prob = self.tournament(pop, fit_pop, prob_pop)

            mutation_prob = p1_fit/(p1_fit+p2_fit)*p1_prob + p2_fit/(p1_fit+p2_fit)*p2_prob  # weighted average dependant on fitness

            c1, c1_prob, c2, c2_prob = self.recombination(mutation_prob, p1, p1_prob, p2, p2_prob)
            c1, c1_prob = self.mutation(mutation_prob, c1, c1_prob)
            c2, c2_prob = self.mutation(mutation_prob, c2, c2_prob)

            all_offspring_weights = np.vstack((all_offspring_weights, c1))
            all_offspring_weights = np.vstack((all_offspring_weights, c2))
            all_offspring_mutation_probs.append(c1_prob)
            all_offspring_mutation_probs.append(c2_prob)

        return (all_offspring_weights, np.array(all_offspring_mutation_probs))

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

    def selection(self, pop, fit_pop, prob_pop) -> tuple[ndarray, ndarray, ndarray]:
        """
        Exponential ranking-based selection with elitism.
        """
        best_i = np.argmax(fit_pop)
        sorted_pop_indices = np.argsort(fit_pop)
        pop_selection_probs = np.array(list(map(lambda i: 1 - np.e**(-1), sorted_pop_indices)))
        pop_selection_probs /= np.sum(pop_selection_probs)
        chosen = np.random.choice(pop.shape[0], POP_SIZE , p=pop_selection_probs, replace=False)
        if best_i not in chosen:
            chosen = np.append(chosen[1:], best_i)  # always include best

        pop = pop[chosen]
        fit_pop = fit_pop[chosen]
        prob_pop = prob_pop[chosen]

        return (pop, fit_pop, prob_pop)

    def evolve_pop(self, pop, fit_pop, prob_pop, reshuffle_t) -> tuple[ndarray, ndarray, ndarray]:
        offspring, prob_offspring = self.reproduce(pop, fit_pop, prob_pop)
        fit_offspring = self.get_fitness(offspring)

        # Select from both parents and offsprong
        alltogether = np.vstack((pop, offspring))
        fit_alltogether = np.append(fit_pop, fit_offspring)
        prob_alltogether = np.append(prob_pop, prob_offspring)

        best_i = np.argmax(fit_alltogether)
        assert fit_alltogether[best_i] == max(fit_alltogether) # TODO remove

        # fit_alltogether[best_i] = float(self.get_fitness(np.array([alltogether[best_i]]))[0])  # repeats best eval, for stability issues TODO literally their code
        assert fit_alltogether[best_i] == max(fit_alltogether) # TODO remove

        self.last_best = alltogether[best_i]
        self.last_best_prob = prob_alltogether[best_i]

        new_pop, new_pop_fit, new_pop_prob = self.selection(alltogether, fit_alltogether, prob_alltogether)
        assert max(fit_alltogether) == max(new_pop_fit) # TODO remove
        # if self.last_best_fit is None or best_fit > self.last_best_fit:
        #     self.last_best_fit = best_fit
        #     self.not_improved = 0
        # else:
        #     self.not_improved += 1

        # if self.not_improved >= reshuffle_t:
        #     new_pop, new_pop_fit = self.reshuffle(new_pop, new_pop_fit)

        return (new_pop, new_pop_fit, new_pop_prob)

    def stats(self, fit_pop, prob_pop):
        """
        TODO very basic, could be better :)
        """
        best_fit = np.max(fit_pop)
        mean_fit = np.mean(fit_pop)
        std_fit = np.std(fit_pop)

        best_prob = np.max(prob_pop)
        mean_prob = np.mean(prob_pop)
        std_prob = np.std(prob_pop)

        print(f"{self.gen}:\tbest_fit:{best_fit}\tmean_fit:{mean_fit}\tstd_fit:{std_fit}\tbest_prob:{best_prob}\tmean_prob:{mean_prob}\tstd_prob:{std_prob}")
        with open(f"{self.env.experiment_name}/stats.csv", "a+") as f:
            f.write(f"{self.gen},{best_fit},{mean_fit},{std_fit},{best_prob},{mean_prob},{std_prob}\n")

    def run_generation(self):
        if self.env.solutions is None:
            pop, fit_pop, prob_pop = self.gen_pop()
        else:
            pop, fit_pop, prob_pop = self.env.solutions

        new_pop, new_fit_pop, new_prob_pop = self.evolve_pop(pop, fit_pop, prob_pop, 10)

        self.stats(new_fit_pop, new_prob_pop)

        self.env.update_solutions([new_pop, new_fit_pop, new_prob_pop])
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
    min_mutation_str = str(MIN_MUTATION).replace(".", "_")
    experiment_name = f"specialized_ea-{min_mutation_str}-{TOURNAMENT_K}-{LOWER_CAUCHY}-{UPPER_CAUCHY}"
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    with open(f"{experiment_name}/stats.csv", "w+") as f:
        f.write("gen,best_fit,mean_fit,std_fit,best_prob,mean_prob:,std_prob\n")

    ea = SpecializedEA(experiment_name, 1)
    for i in range(GENERATIONS):
        ea.run_generation()
        # if i % 10 == 0:
        #     ea.show_best()

    ea.show_best()
