import pandas as pd
import numpy as np
import random
from tqdm import tqdm
def initialize_population(num_individuals, num_cities):
    population = []
    for _ in range(num_individuals):
        individual = list(range(1, num_cities + 1))
        random.shuffle(individual)
        population.append(individual)
    return population

def calculate_fitness(individual, distances):
    total_distance = 0
    for i in range(len(individual) - 1):
        total_distance += distances[individual[i] - 1][individual[i + 1] - 1]
    total_distance += distances[individual[-1] - 1][individual[0] - 1]  # Return to the starting city
    return total_distance

# def select_parents(population, distances):
#     fitness_scores = [1 / calculate_fitness(individual, distances) for individual in population]
#     total_fitness = sum(fitness_scores)
#     probabilities = [score / total_fitness for score in fitness_scores]
#     parents = np.random.Generator.choice(population, size=2, p=probabilities, replace=False)
#     return parents
def select_parents(population, distances):
    fitness_scores = [1 / calculate_fitness(individual, distances) for individual in population]
    total_fitness = sum(fitness_scores)
    probabilities = [score / total_fitness for score in fitness_scores]

    # Wybierz indeksy rodziców
    parents_indices = np.random.choice(len(population), size=2, p=probabilities, replace=False)

    # Wybierz dwuwymiarowych rodziców z populacji
    parents = [population[idx] for idx in parents_indices]

    return parents
# def select_parents(population, distances):
#     fitness_scores = [1 / calculate_fitness(individual, distances) for individual in population]
#     total_fitness = sum(fitness_scores)
#     probabilities = [score / total_fitness for score in fitness_scores]
#
#     # Spłaszcz listę populacji
#     flat_population = [item for sublist in population for item in sublist]
#
#     # Wybierz indeksy rodziców
#     parents_indices = np.random.choice(len(flat_population), size=2, p=probabilities, replace=False)
#
#     # Przekształć indeksy na pary indeksów odpowiadające dwóm rodzicom w dwuwymiarowej populacji
#     parents_indices_2d = [(idx // len(flat_population[0]), idx % len(flat_population[0])) for idx in parents_indices]
#
#     # Wybierz rodziców z dwuwymiarowej populacji
#     parents = [[flat_population[i][j] for i, j in parents_indices_2d[0]],
#                [flat_population[i][j] for i, j in parents_indices_2d[1]]]
#
#     return parents

def crossover(parents):
    crossover_point = random.randint(1, len(parents[0]) - 1)
    child1 = parents[0][:crossover_point] + [city for city in parents[1] if city not in parents[0][:crossover_point]]
    child2 = parents[1][:crossover_point] + [city for city in parents[0] if city not in parents[1][:crossover_point]]
    return child1, child2

def mutate(individual, mutation_rate):
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(len(individual)), 2)
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual

def genetic_algorithm(distances, population_size=100, generations=100, mutation_rate=0.01):
    num_cities = len(distances)
    population = initialize_population(population_size, num_cities)

    for generation in tqdm(range(generations)):
        new_population = []
        for _ in range(population_size // 2):
            parents = select_parents(population, distances)
            offspring1, offspring2 = crossover(parents)
            offspring1 = mutate(offspring1, mutation_rate)
            offspring2 = mutate(offspring2, mutation_rate)
            new_population.extend([offspring1, offspring2])

        population = new_population

    best_individual = min(population, key=lambda ind: calculate_fitness(ind, distances))
    best_fitness = calculate_fitness(best_individual, distances)
    return best_individual, best_fitness

# Read distances from the provided Excel file
excel_file = 'Przykład_TSP_29.xlsx'
distances_df = pd.read_excel(excel_file, index_col=0)
distances = distances_df.values

# Run the genetic algorithm
best_route, best_distance = genetic_algorithm(distances)

print("Best Route:", best_route)
print("Best Distance:", best_distance)

