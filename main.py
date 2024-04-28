import copy
from tqdm import tqdm

import numpy as np

POPULATION_SIZE = 10000  # population size
FUNCTION_DOMAIN = [-5.0, 5]  # genes range
PARAMETER_COUNT = 2  # no. of parameters
MUTATION_RATE = 0.1  # rate of mutations
SEED = 42
show_progress_bar = True

expected_value_single = -39.1661657037714
TARGET = PARAMETER_COUNT * expected_value_single  # goal

# ----- CONFIG -----
rng = np.random.default_rng(SEED)


class Genome:
    def __init__(self):
        self.chromo = []
        self.fitness = np.inf

    def generate(self):
        self.chromo = rng.uniform(FUNCTION_DOMAIN[0], FUNCTION_DOMAIN[1], PARAMETER_COUNT)
        self.fitness = float(styblinski_tang(self.chromo) - TARGET)

    def set_chromo(self, chromo):
        self.chromo = chromo
        self.fitness = float(styblinski_tang(self.chromo) - TARGET)

    def mutate(self):
        for i in range(PARAMETER_COUNT):
            if rng.random() < MUTATION_RATE:
                self.chromo[i] = float(rng.uniform(FUNCTION_DOMAIN[0], FUNCTION_DOMAIN[1], 1))

    def crossover(self, individual):
        exchange_point = rng.integers(low=0, high=POPULATION_SIZE - 1, size=1)
        new_chromo = Genome()
        p1 = list(self.chromo[:int(exchange_point)])
        p2 = list(individual.chromo[int(exchange_point):])
        new_chromo.set_chromo(p1+p2)
        return new_chromo


def styblinski_tang(value_array):
    result = 0.0
    for val in value_array:
        result += val ** 4 - 16 * val ** 2 + 5 * val
    return result / 2


def crossover(better_half, population):
    crossed = list()
    for i in tqdm(range(POPULATION_SIZE), disable=not show_progress_bar, desc="Crossover"):
        parent1 = rng.choice(better_half)
        parent2 = rng.choice(population)
        crossed.append(parent1.crossover(parent2))  # Parents are crossing and result is appended to crossed list
    return crossed


def mutate(crossed):
    print("Mutate stage")
    mutated_offspring = []
    for genome in tqdm(crossed, disable=not show_progress_bar, desc="Mutation"):
        genome.mutate()
        mutated_offspring.append(genome)
    return mutated_offspring


def select_tournament(population):
    """
    Best of 3 tournament
    """
    best_population = list()
    for _ in tqdm(range(POPULATION_SIZE//2), disable=not show_progress_bar, desc="Tournament"):
        tournament = list()
        for i in range(3):
            tournament.append(rng.choice(population))
        tournament.sort(key=lambda x: x.fitness)
        best_population.append(tournament[0])
    return best_population


def initialize_population(instance_size):
    print("Initializing population stage")
    population = list()
    for i in range(instance_size):
        genome = Genome()
        genome.generate()
        population.append(genome)
    return population


def replace(new_gen, population):
    print("Replace stage")
    for _ in tqdm(range(POPULATION_SIZE), disable=not show_progress_bar, desc="Replace"):
        if population[_].fitness > new_gen[_].fitness:
            population[_] = new_gen[_]
    return population


"""
    Genetic algorithm steps:
    1. Initialize a population N
    2. Select individuals to evolve from
    3. Crossover of couples of parents
    4. Mutate individuals according to their mutation rate
"""

def main():
    initial_population = initialize_population(POPULATION_SIZE)
    found = False
    population = initial_population
    generation = 1

    print("Initial Population Size: ", POPULATION_SIZE)
    initial_population.sort(key=lambda x: x.fitness)
    print(f"Values: {initial_population[0].chromo}, Fitness: {initial_population[0].fitness} Generation {generation}")

    while not found:
        best_half = select_tournament(population)
        population = sorted(population, key=lambda x: x.fitness)
        crossed = crossover(best_half, population)
        new_population = mutate(crossed)

        population = replace(new_population, population)

        if population[0].fitness == 0:
            print("Target found!")
            print(f"Values: {population[0].chromo}, Generation {generation}")
            break
        print(f"Values: {population[0].chromo}, Fitness: {population[0].fitness} Generation {generation}, Population size: {population.__len__()}")
        generation += 1


if __name__ == "__main__":
    main()
