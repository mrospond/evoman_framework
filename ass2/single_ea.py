import random
import numpy as np

from demo_controller import player_controller
from evoman.environment import Environment
from numpy import ndarray

from global_params import *


class SingleEA():
    def __init__(
        self, id: int, experiment_name: str, enemies: list[int], pop_size: int,
        pop: ndarray | None = None, communicates: bool = False
    ) -> None:
        self.id = id

        self.env = Environment(
            experiment_name=experiment_name,
            enemies=enemies,
            multiplemode="yes",
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
        self.communicates = communicates  # True if EA is an island, False otherwise
        self.pop_size = pop_size

        # Initialize population
        if pop is None:
            (self.pop, self.pop_fit) = self.gen_pop(self.pop_size)
        else:
            # assert len(pop) == self.pop_size TODO
            self.pop = pop
            self.pop_fit = self.get_fitness(pop)

    def simulate(self, pop) -> float:
        fitness, playerlife, enemylife, time = self.env.play(pcont=pop)
        return fitness

    def get_fitness(self, pop) -> ndarray:
        return np.array([self.simulate(indiv) for indiv in pop])

    def gen_pop(self, pop_size: int) -> tuple[ndarray, ndarray]:
        """
        Generates pop_size random individuals.
        Returns the new population and their fitness.
        """
        pop = np.random.uniform(LIM_LOWER, LIM_UPPER, (pop_size, self.n_weights))
        fit_pop = self.get_fitness(pop)

        return (np.array(pop), fit_pop)

    def parent_selection(self) -> ndarray:
        """
        TODO: decide on parent selection method.
        Tournament with k=2.
        Sample: no replacement
        """
        sample = random.sample(range(len(self.pop)), TOURNAMENT_K)  # generate indexes of sampled individuals

        best_fit = self.pop_fit[sample[0]]
        best_i = sample[0]
        for i in sample:
            if self.pop_fit[i] > best_fit:
                best_fit = self.pop_fit[i]
                best_i = i

        return self.pop[best_i]

    def recombination(self, p1: ndarray, p2: ndarray) -> tuple[ndarray, ndarray]:
        """
        TODO: decide on recombination method.
        n-point crossover with n=2.
        """
        if random.uniform(0, 1) <= CROSSOVER_PROB:  # TODO: for pair or indiv?
            # Crossover points
            cp1 = random.choice(range(len(p1)-1))
            cp2 = random.choice(range(cp1+1, len(p1)))

            c1 = np.append(np.copy(p1[0:cp1]), np.append(np.copy(p2[cp1:cp2]), np.copy(p1[cp2:])))
            c2 = np.append(np.copy(p2[0:cp1]), np.append(np.copy(p1[cp1:cp2]), np.copy(p2[cp2:])))
        else:
            c1 = np.copy(p1)
            c2 = np.copy(p2)

        return (c1, c2)

    def mutation(self, indiv: ndarray) -> ndarray:
        """
        Mutate the individual weights of indiv. For each weight, a new value is
        generated that determines whether it will be mutated, and if so, a
        new value from a TODO distribution is drawn.
        """
        for i in range(len(indiv)):
            if random.uniform(0, 1) <= MUTATION_PROB:
                delta = random.uniform(-2, 2)
                new_val = LIM_LOWER + (indiv[i] + delta + LIM_LOWER) % (LIM_UPPER-LIM_LOWER)
                indiv[i] = min(LIM_UPPER, max(LIM_LOWER, new_val))

        return indiv

    def reproduce(self):
        all_offspring_weights = np.empty((0, self.n_weights))
        for _ in range(0, POP_SIZE, 2):  # go by pairs of parents
            p1 = self.parent_selection()
            p2 = self.parent_selection()

            c1, c2 = self.recombination(p1, p2)
            c1 = self.mutation(c1)
            c2 = self.mutation(c2)

            all_offspring_weights = np.vstack((all_offspring_weights, c1))
            all_offspring_weights = np.vstack((all_offspring_weights, c2))

        return all_offspring_weights

    def survivor_selection(self, pop: ndarray, pop_fit: ndarray) -> tuple[ndarray, ndarray]:
        """
        TODO: decide survivor selection method.
        """
        survivor_is = random.sample(range(len(pop)), self.pop_size)
        survivors = np.zeros((0, self.n_weights))
        for i in survivor_is:
            survivors = np.vstack((survivors, pop[i]))
        survivors_fit = self.get_fitness(survivors)

        return (survivors, survivors_fit)

    def evolve_pop(self) -> tuple[ndarray, ndarray]:
        offspring = self.reproduce()
        offspring_fit = self.get_fitness(offspring)

        # Select from both parents and offspring TODO
        alltogether = np.vstack((self.pop, offspring))
        fit_alltogether = np.append(self.pop_fit, offspring_fit)
        survivors, survivors_fit = self.survivor_selection(alltogether, fit_alltogether)

        return (survivors, survivors_fit)

    def stats(self, fit_pop):
        """
        TODO very basic, could be better :)
        """
        best_fit = np.max(fit_pop)  # self.last_best_fit
        mean_fit = np.mean(fit_pop)
        std_fit = np.std(fit_pop)

        print(f"{self.gen},{self.id}:\tenemies:{self.env.enemies}\tbest_fit:{best_fit}\tmean_fit:{mean_fit}\tstd_fit:{std_fit}")
        enemies_str = "".join([str(e) for e in self.env.enemies])
        with open(f"{self.env.experiment_name}/stats.csv", "a+") as f:  # TODO: parallel
            f.write(f"{self.gen},{self.id},{enemies_str},{best_fit},{mean_fit},{std_fit}\n")

    def run_generation(self):
        survivors, survivors_fit = self.evolve_pop()

        self.stats(survivors)

        self.pop = survivors
        self.pop_fit = survivors_fit

        self.env.update_solutions([survivors, survivors_fit])

        self.gen += 1

    def emigration(self) -> ndarray:
        """
        Returns NUM_MIGRATION individuals according to TODO selection method
        to be copied to other EAs during migration.
        """
        migrant_is = random.sample(range(len(self.pop)), NUM_MIGRATION)
        migrants = np.zeros((0, self.n_weights))
        for i in migrant_is:
            migrants = np.vstack((migrants, self.pop[i]))

        return migrants

    def immigration(self, migrants: ndarray) -> None:
        """
        Copy migrants into population.
        """
        for indiv in migrants:
            self.pop = np.vstack((self.pop, np.copy(indiv)))
            self.pop_fit = np.append(self.pop_fit, self.get_fitness([indiv]))
            assert len(self.pop) > 0
