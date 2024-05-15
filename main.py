import copy
from tqdm import tqdm
from time import sleep
import numpy as np

POPULATION_SIZE = 5000  # population size
FUNCTION_DOMAIN = [-5.0, 5.0]  # genes range
PARAMETER_COUNT = 5  # no. of parameters
MUTATION_RATE = 0.05  # rate of mutations
SEED = 1945
show_progress_bar = True
leave_progress_bar = False

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
        new_chromo = Genome()
        p1 = list(self.chromo[:int(exchange_point)])
        p2 = list(individual.chromo[int(exchange_point):])
        new_chromo.set_chromo(p1+p2)
        return new_chromo


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


def crossover(better_half, population):
    # 70% - 90% prawdopodobienstwo krzyżowania!
    """
    Crossover of the better half of the population with the rest of the population

    :param better_half: better half of the population
    :param population: population to crossover with
    :return: list of crossed genomes
    """
    crossed = list()
    for i in tqdm(range(POPULATION_SIZE), disable=not show_progress_bar, desc="Crossover", leave=leave_progress_bar):
        parent1 = copy.deepcopy(rng.choice(better_half))
        parent2 = copy.deepcopy(rng.choice(population))
        crossed.append(parent1.crossover(parent2))  # Parents are crossing and result is appended to crossed list
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
    initial_population = initialize_population(POPULATION_SIZE)
    found = False
    population = initial_population
    generation = 0

    print("Initial Population Size: ", POPULATION_SIZE)
    initial_population.sort(key=lambda x: x.fitness)
    best_fitness = initial_population[0].fitness
    print(f"Best fitness: {best_fitness}")

    # Do the loop until the target is found
    while not found:
        sleep(0.1)
        tournament_winners = select_tournament(population)
        population = sorted(population, key=lambda x: x.fitness)
        crossed = crossover(tournament_winners, population) # nie wybierać z populacji!!
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


if __name__ == "__main__":
    main()
