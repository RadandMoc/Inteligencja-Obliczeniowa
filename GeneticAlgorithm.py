import pandas as pd
import numpy as np
import random
from tqdm import tqdm


#Zwykly
"""

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
def select_parents(population,probabilities):
   

    # Wybierz indeksy rodziców
    parents_indices = np.random.choice(len(population), size=2, p=probabilities, replace=False)

    # Wybierz dwuwymiarowych rodziców z populacji
    parents = [population[idx] for idx in parents_indices]

    return parents

def get_probabilites_of_crossover(population,distances):
    fitness_scores = [1 / calculate_fitness(individual, distances) for individual in population]
    total_fitness = sum(fitness_scores)
    probabilities = [ score / total_fitness for score in fitness_scores]
    
    return probabilities
    


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

def genetic_algorithm(distances, population_size=100, generations=100, mutation_rate=0.01, number_of_crossover = 900, elite_percent = 0.1):
    num_cities = len(distances)
    population = initialize_population(population_size, num_cities)
    
    for generation in tqdm(range(generations)):
        children = []
        probabilities = get_probabilites_of_crossover(population,distances)

        for _ in range( number_of_crossover // 2):
            parents = select_parents(population, probabilities)
            offspring1, offspring2 = crossover(parents)
            offspring1 = mutate(offspring1, mutation_rate)
            offspring2 = mutate(offspring2, mutation_rate)
            children.extend([offspring1, offspring2])
        
        

        population.sort(key=lambda individual: calculate_fitness(individual, distances))
        elite_count = int(elite_percent*population_size)
        top_previous = population[:elite_count]

        children.sort(key=lambda individual: calculate_fitness(individual, distances))
        top_children = children[:population_size-elite_count]


        population = top_previous + top_children

    best_individual = min(population, key=lambda ind: calculate_fitness(ind, distances))
    best_fitness = calculate_fitness(best_individual, distances)
    return best_individual, best_fitness
"""







def initialize_population(num_individuals, num_cities):
    # No change needed here, the function is already efficient
    return np.array([np.random.permutation(num_cities) + 1 for _ in range(num_individuals)])

def calculate_fitness_vectorized(population, distances):
    # Vectorized fitness calculation
    indices = np.array(population) - 1  # Adjust indices for 0-based indexing
    rolled_indices = np.roll(indices, -1, axis=1)  # Roll indices to get pairs for distance calculation
    total_distances = distances[indices, rolled_indices].sum(axis=1)
    return total_distances

def tournament_selection(population, fitness_scores, tournament_size=3):
    # Select parents using tournament selection
    selected_indices = []
    for _ in range(2):
        participants = np.random.choice(len(population), tournament_size, replace=False)
        winner_idx = participants[np.argmin(fitness_scores[participants])]
        selected_indices.append(winner_idx)
    return population[selected_indices]

def roulette_wheel_selection(population, fitness_scores):
    # Select parents using roulette wheel selection
    total_fitness = sum(fitness_scores)
    probabilities = fitness_scores/total_fitness
    participants = np.random.choice(len(population), size=2, p=probabilities, replace=False)
    return population[participants]

def crossover1(parents):
    # No change needed here, the function is efficient
    crossover_point = random.randint(1, len(parents[0]) - 1)
    child1 = np.concatenate([parents[0][:crossover_point], 
                             [city for city in parents[1] if city not in parents[0][:crossover_point]]])
    child2 = np.concatenate([parents[1][:crossover_point], 
                             [city for city in parents[0] if city not in parents[1][:crossover_point]]])
    return child1, child2

def crossover2(parents):
    number_of_cities = len(parents[1])
    crossover_point_1 = round(number_of_cities/3)
    crossover_point_2 = round(number_of_cities*(2/3))
    child1 = []
    child2 = []
    
    count = 0
    for i in parents[0]:
        if(count == crossover_point_1):
            break
        if(i not in parents[1][crossover_point_1:crossover_point_2]):
            child1.append(i)
            count = count+1       
    child1.extend(parents[1][crossover_point_1:crossover_point_2])
    child1.extend([city for city in parents[0] if city not in child1])

    count = 0
    for i in parents[1]:
        if(count == crossover_point_1):
            break
        if(i not in parents[0][crossover_point_1:crossover_point_2]):
            child2.append(i)
            count = count+1      
    child2.extend(parents[0][crossover_point_1:crossover_point_2])
    child2.extend([city for city in parents[1] if city not in child2])
    return child1, child2

def mutate(individual, mutation_rate):
    # No change needed here, the function is efficient
    if random.random() < mutation_rate:
        idx1, idx2 = np.random.choice(len(individual), 2, replace=False)
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual

# Optimized Genetic Algorithm

def genetic_algorithm(distances, population_size=100, generations=100, mutation_rate=0.01, number_of_crossover=900, elite_percent=0.1, iteration_without_improvement=20):
    num_cities = distances.shape[0]
    population = initialize_population(population_size, num_cities)
    fitness_scores = calculate_fitness_vectorized(population, distances)

    # Variables needed for stopping criterion
    best_index_of_first_population = np.argmin(fitness_scores)
    first_best_score = fitness_scores[best_index_of_first_population]
    number_without_improvement = 0

    for generation in tqdm(range(generations)):
        # Tournament selection for parents
        children = np.empty((number_of_crossover, num_cities), dtype=int)
        for i in range(0, number_of_crossover, 2):
            parents = tournament_selection(population, fitness_scores)
            offspring = crossover1(parents)
            children[i] = mutate(offspring[0], mutation_rate)
            children[i+1] = mutate(offspring[1], mutation_rate)

        # Recalculate fitness for the new children
        children_fitness = calculate_fitness_vectorized(children, distances)

        # Elitism: Keep the best individuals from the current population
        elite_indices = np.argsort(fitness_scores)[:int(elite_percent * population_size)]
        elite_population = population[elite_indices]

        # Combine elite individuals and best children to form new population
        top_children_indices = np.argsort(children_fitness)[:population_size - len(elite_population)]
        population = np.vstack([elite_population, children[top_children_indices]])

        # Fitness scores for generation
        fitness_scores = calculate_fitness_vectorized(population, distances)
        # Best index of generation
        best_index_of_generation = np.argmin(fitness_scores)
         
        # Stopping criterion implementation - checking number of iteration without improvement
        # If it's first generation
        if(generation==0):
            best_score_of_pregeneration = fitness_scores[best_index_of_generation]
            if(fitness_scores[best_index_of_generation]<first_best_score): 
                number_without_improvement = 0
            else:
                number_without_improvement += 1 
        # In other options
        else:
            if(fitness_scores[best_index_of_generation]<best_score_of_pregeneration): 
                number_without_improvement = 0
            else:
                number_without_improvement += 1 
            if(number_without_improvement == iteration_without_improvement):
                break
        # Save best score of generation for the next checking 
        best_score_of_pregeneration = fitness_scores[best_index_of_generation]

    # Find the best solution in the final population
    best_index = np.argmin(fitness_scores)
    return population[best_index], fitness_scores[best_index]













# Read distances from the provided Excel file
readData=pd.read_csv("Dane_TSP_127.csv",sep=";", decimal=',')
#readData=pd.read_csv("Dane_TSP_76.csv",sep=";", decimal=',')

#readData=pd.read_csv("Miasta29.csv",sep=";", decimal=',')

distance_matrix = readData.iloc[:,1:].astype(float).to_numpy()


#print(crossover([[12, 18, 17, 14, 15, 26, 3, 6, 8, 28, 27, 23, 25, 7, 11, 22, 10, 24, 19, 4, 16, 9, 21, 2, 20, 5, 1, 13, 29],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,21,23,24,25,26,27,28,29]]))
# Run the genetic algorithm
best_route, best_distance = genetic_algorithm(distance_matrix,50,generations=10000,mutation_rate=0.05, number_of_crossover = 250, elite_percent=0.1, iteration_without_improvement=100)

print("Best Route:", best_route)
print("Best Distance:", best_distance)

