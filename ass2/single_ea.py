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
            assert len(pop) == self.pop_size
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

    def parent_selection(self) -> tuple[ndarray, ndarray]:
        """
        TODO: decide on parent selection method.
        Sample: no replacement
        """
        parent_indeces = random.sample(range(len(self.pop)), 2)
        return (self.pop[parent_indeces[1]], self.pop[parent_indeces[1]])

    def recombination(self, p1: ndarray, p2: ndarray) -> tuple[ndarray, ndarray]:
        """
        TODO: decide on recombination method.
        """
        if random.uniform(0, 1) <= CROSSOVER_PROB:  # TODO: for pair or indiv?
            c1 = np.copy(p1)
            c2 = np.copy(p2)
            # TODO
        else:
            c1 = np.copy(p1)
            c2 = np.copy(p2)

        return (c1, c2)

    def mutation(self, indiv: ndarray) -> ndarray:
        if random.uniform(0, 1) <= MUTATION_PROB:
            # TODO: mutate
            pass

        return indiv

    def reproduce(self):
        all_offspring_weights = np.empty((0, self.n_weights))
        for _ in range(0, POP_SIZE, 2):  # go by pairs of parents
            p1, p2 = self.parent_selection()

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
            self.pop_fit = np.vstack((self.pop_fit, self.get_fitness([indiv])))
