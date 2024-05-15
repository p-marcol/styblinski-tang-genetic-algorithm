import copy
import os.path

from tqdm import tqdm
from time import sleep
import numpy as np
import signal
import sys
import datetime

POPULATION_SIZE = 5000  # population size
FUNCTION_DOMAIN = [-5.0, 5.0]  # genes range
PARAMETER_COUNT = 5  # no. of parameters
MUTATION_RATE = 0.05  # rate of mutations
CROSS_RATE = 0.7  # rate of crossover (0.7 - 0.9)
SEED = 1945
show_progress_bar = True
leave_progress_bar = False
BEST_INDIVIDUAL_STR = ""
expected_value_single = -39.1661657037714
TARGET = PARAMETER_COUNT * expected_value_single  # goal

# ----- CONFIG -----
rng = np.random.default_rng(SEED)


class Genome:
    """
    Genome class represents a single individual in the population
    """
    def __init__(self):
        """
        Initialize the genome with an empty chromosome and infinite fitness
        """
        self.chromo = []
        self.fitness = np.inf

    def generate(self):
        """
        Generate a random chromosome using the rng object and calculate the fitness
        """
        self.chromo = rng.uniform(FUNCTION_DOMAIN[0], FUNCTION_DOMAIN[1], PARAMETER_COUNT)
        self.fitness = float(styblinski_tang(self.chromo) - TARGET)

    def set_chromo(self, chromo):
        """
        Set the chromosome passed as an argument and calculate the fitness

        :param chromo: new chromosome to set
        """
        self.chromo = chromo
        self.fitness = float(styblinski_tang(self.chromo) - TARGET)

    def mutate(self):
        """
        Mutate the genome with a probability of MUTATION_RATE
        """
        for i in range(PARAMETER_COUNT):
            if rng.random() < MUTATION_RATE:
                self.chromo[i] = float(rng.uniform(FUNCTION_DOMAIN[0], FUNCTION_DOMAIN[1]))

    def crossover(self, individual):
        """
        Crossover the genome with another genome passed as an argument

        :param individual: genome to crossover with
        """
        exchange_point = rng.integers(low=0, high=PARAMETER_COUNT)
        child1 = Genome()
        child2 = Genome()

        parent1left = list(self.chromo[:int(exchange_point)])
        parent2right = list(individual.chromo[int(exchange_point):])

        parent2left = list(individual.chromo[:int(exchange_point)])
        parent1right = list(self.chromo[int(exchange_point):])

        child1.set_chromo(parent1left+parent2right)
        child2.set_chromo(parent2left+parent1right)
        return [child1, child2]

    def __str__(self):
        return str(self.chromo)


def styblinski_tang(value_array):
    """
    Method to calculate he value of the Styblinski-Tang function. Result is used to calculate the fitness of the genome

    :param value_array: array that contains the values of the parameters
    :return: result of the Styblinski-Tang function
    """
    result = 0.0
    for val in value_array:
        result += val ** 4 - 16 * val ** 2 + 5 * val
    return result / 2


def crossover(population):
    crossed = list()
    for i in tqdm(range(POPULATION_SIZE), disable=not show_progress_bar, desc="Crossover", leave=leave_progress_bar):
        parent1 = copy.deepcopy(rng.choice(population))
        parent2 = copy.deepcopy(rng.choice(population))
        if rng.random() < CROSS_RATE:
            children = parent1.crossover(parent2)
            crossed.extend(children)
        else:
            crossed.append(parent1)
            crossed.append(parent2)
    return crossed


def mutate(crossed):
    """
    Mutation of the crossed genomes. Each genome has a chance of being mutated

    :param crossed: list of genomes to mutate
    :return: list of all genomes including the mutated ones
    """
    mutated_offspring = list()
    for genome in tqdm(crossed, disable=not show_progress_bar, desc="Mutation", leave=leave_progress_bar):
        genome.mutate()
        mutated_offspring.append(genome)
    return mutated_offspring


def select_tournament(population):
    """
    Tournament selection of the best individuals (best of 3) from the population

    :param population: population to select the best individuals from
    :return: list of the best individuals
    """

    # Każdy osobnik może zostać wybrany maksymalnie POPULATION_SIZE//2 razy.
    # Czy osobniki nie powinny być usuwane po wyborze?
    best_population = list()
    for _ in tqdm(range(POPULATION_SIZE//2), disable=not show_progress_bar, desc="Tournament", leave=leave_progress_bar):
        tournament = list()
        for i in range(3):
            tournament.append(rng.choice(population))
        tournament.sort(key=lambda x: x.fitness)
        best_population.append(tournament[0])
    return best_population


def initialize_population(instance_size):
    """
    Initialize the population with random genomes

    :param instance_size: size of the population to initialize
    :return: new population
    """
    print("Initializing population stage")
    population = list()
    for i in range(instance_size):
        genome = Genome()
        genome.generate()
        population.append(genome)
    return population


def replace(new_gen, population):
    """
    Join both populations and sort them by fitness. Return the better half of the population

    :param new_gen: new population
    :param population: old population
    :return: better half of the population
    """
    sorted = (new_gen + population)
    sorted.sort(key=lambda x: x.fitness)
    return sorted[:POPULATION_SIZE]

    # Genetic algorithm steps:
    # 1. Initialize a population N
    # 2. Select individuals to evolve from
    # 3. Crossover of couples of parents
    # 4. Mutate individuals according to their mutation rate


def main():
    """
    Main function that runs the genetic algorithm

    Genetic algorithm steps:
    1. Initialize a population N
    2. Select individuals to evolve from
    3. Crossover of couples of parents
    4. Mutate individuals according to their mutation rate

    """
    signal.signal(signal.SIGINT, signal_handler)
    initial_population = initialize_population(POPULATION_SIZE)
    found = False
    population = initial_population
    generation = 0

    print("Initial Population Size: ", POPULATION_SIZE)
    initial_population.sort(key=lambda x: x.fitness)
    best_fitness = initial_population[0].fitness
    global BEST_INDIVIDUAL_STR
    BEST_INDIVIDUAL_STR = str(population[0])
    print(f"Best fitness: {best_fitness}")

    # Do the loop until the target is found
    while not found:
        sleep(0.1)
        tournament_winners = select_tournament(population)
        population = sorted(population, key=lambda x: x.fitness)
        crossed = crossover(tournament_winners)
        new_population = mutate(crossed)

        population = replace(new_population, population)

        generation += 1
        if population[0].fitness == 0:
            print("Target found!")
            print(f"Values: {population[0].chromo}, Generation {generation}")
            break
        if population[0].fitness < best_fitness:
            best_fitness = population[0].fitness
            print(f"New best fitness! {best_fitness}")
        print(f"G{generation}:\tBest values: {population[0].chromo}, Fitness: {population[0].fitness}")


def signal_handler(sig, frame):
    if not os.path.exists("logs"):
        os.makedirs("logs")
    filename = "logs/best_" + str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) + ".log"
    logfile = open(filename, mode="x")
    logfile.write(BEST_INDIVIDUAL_STR)
    logfile.close()
    print("\nClosing program. The best individual is saved in " + filename + "\n")
    sys.exit(0)


if __name__ == "__main__":
    main()
