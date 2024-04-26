import numpy as np
from random import seed, uniform, random

POPULATION_SIZE = 500           # population size
FUNCTION_DOMAIN = [-5.0, 5]     # genes range
PARAMETER_COUNT = 2             # no. of parameters
MUTATION_RATE = 0.01            # rate of mutations
SEED = 42

expected_value_single = -39.1661657037714
TARGET = PARAMETER_COUNT * expected_value_single    # goal

# ----- CONFIG -----
# seed(10)
# frand = generator.uniform(FUNCTION_DOMAIN[0], FUNCTION_DOMAIN[1])
# frand = random() * (FUNCTION_DOMAIN[1] + FUNCTION_DOMAIN[0]) - FUNCTION_DOMAIN[0]
rng = np.random.default_rng(SEED)


def styblinski_tang(value_array):
    result = 0.0
    for val in value_array:
        result += val ** 4 - 16 * val ** 2 + 5 * val
    return result / 2


def fitness_calc(population_chromo):
    return [population_chromo, styblinski_tang(population_chromo) - TARGET]


def select_best_half(population):
    population.sort(key=lambda x: fitness_calc(x)[1], reverse=False)
    return population[:(POPULATION_SIZE // 2)]


def initialize_population():
    population = list()
    for i in range(POPULATION_SIZE):
        population.append(rng.uniform(FUNCTION_DOMAIN[0], FUNCTION_DOMAIN[1], PARAMETER_COUNT))
    return population


def main():
    print("Hello, World!")
    print(initialize_population()[:5])


if __name__ == "__main__":
    main()
