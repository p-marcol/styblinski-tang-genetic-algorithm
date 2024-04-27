import numpy as np
from random import seed, uniform, random

POPULATION_SIZE = 100  # population size
FUNCTION_DOMAIN = [-5.0, 5]  # genes range
PARAMETER_COUNT = 2  # no. of parameters
MUTATION_RATE = 0.01  # rate of mutations
SEED = 42

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
        self.fitness = styblinski_tang(self.chromo) - TARGET

    def set_chromo(self, chromo):
        self.chromo = chromo
        self.fitness = styblinski_tang(self.chromo) - TARGET

    def mutate(self):
        for i in range(PARAMETER_COUNT):
            if rng.random() < MUTATION_RATE:
                self.chromo[i] = rng.uniform(FUNCTION_DOMAIN[0], FUNCTION_DOMAIN[1], 1)

    def crossover(self, individual):
        exchange_point = rng.integers(low=0, high=POPULATION_SIZE - 1, size=1)
        new_chromo = Genome()
        new_chromo.set_chromo(np.concatenate((self.chromo[:int(exchange_point)], individual.chromo[int(exchange_point):]), axis=None))
        return new_chromo



def styblinski_tang(value_array):
    result = 0.0
    for val in value_array:
        result += val ** 4 - 16 * val ** 2 + 5 * val
    return result / 2


def crossover(better_half, population):
    crossed = []
    for i in range(POPULATION_SIZE):
        parent1 = rng.choice(better_half)
        parent2 = rng.choice(population[:(POPULATION_SIZE // 2)])
        crossed.append(parent1.crossover(parent2))
    return crossed


def mutate(exchanged):
    mutated_offspring = []
    for genome in exchanged:
        genome.mutate()
        mutated_offspring.append(genome)
    return mutated_offspring


def fitness_calc(population_chromo):
    difference = abs(styblinski_tang(population_chromo) - TARGET)
    return [population_chromo, difference]


def select_best_half(population):
    sorted_population = sorted(population, key=lambda x: x.fitness, reverse=False)
    return sorted_population[:(POPULATION_SIZE // 2)]


def initialize_population(instance_size):
    population = list()
    for i in range(instance_size):
        genome = Genome()
        genome.generate()
        population.append(genome)
    return population


def replace(new_gen, population):
    for _ in range(POPULATION_SIZE):
        if population[_].fitness > new_gen[_].fitness:
            population[_] = new_gen[_]
    return population


def main():
    initial_population = initialize_population(POPULATION_SIZE)
    found = False
    population = initial_population
    generation = 1

    while not found:
        best_half = select_best_half(population)
        population = sorted(population, key=lambda x: x.fitness, reverse=False)
        crossed = crossover(best_half, population)
        new_population = mutate(crossed)

        population = replace(new_population, population)

        if population[0].fitness == 0:
            print("Target found!")
            print(f"Values: {population[0].chromo}, Generation {generation}")
            break
        print(f"Values: {population[0].chromo}, Fitness: {population[0].fitness} Generation {generation}")
        generation += 1


if __name__ == "__main__":
    main()
