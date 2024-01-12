import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import datetime


def saveData(best_route, distance, population_size, generation_size ,number_of_crossover, mutation, iteration_without_improvement,filename = "genetic_algorithm_results.txt" ):
        with open(filename, 'a') as resultFile:
            resultFile.write("\n" + "=" * 25 + "\n")
            for element in best_route:
                resultFile.write(str(element) + ' ')
            resultFile.write("\n" + "Najlepsza odleglosc: " + str(distance))
            resultFile.write("\n" + "Generacji " + str(generation_size))
            resultFile.write("\n + Rozmiar populacji" + str(population_size))
            resultFile.write("\n + mutacja" + str(mutation))
            resultFile.write("\n" + "Liczba iteracji bez poprawy " + str(iteration_without_improvement))
            resultFile.write("\n" + "Liczba Krzyżówek: " + str(number_of_crossover))
            resultFile.write("\n" + "Populacja Początkowa: " + str(population_size))

def save_the_best_individual(individual, total_route_length, generation ,filename, average_route):
    with open(filename, 'a') as file:
        for element in individual:
            file.write(str(element+1) + ' ')
        file.write(f"\n Całkowita Długość trasy dla najlepszego osobnika generacji {generation} to {total_route_length}\n")
        file.write(f"Srednia suma dystansow {average_route}\n")


def initialize_population(num_individuals, num_cities):
    return np.array([np.random.permutation(num_cities) + 1 for _ in range(num_individuals)])

def calculate_fitness_vectorized(population, distances):
    indices = np.array(population) - 1  
    rolled_indices = np.roll(indices, -1, axis=1)
    total_distances = distances[indices, rolled_indices].sum(axis=1)
    return total_distances

def tournament_selection(population, fitness_scores, tournament_size=3):
    selected_indices = []
    for _ in range(2):
        participants = np.random.choice(len(population), tournament_size, replace=False)
        winner_idx = participants[np.argmin(fitness_scores[participants])]
        selected_indices.append(winner_idx)
    return population[selected_indices]

def roulette_wheel_selection(population, fitness_scores):
    total_fitness = sum(fitness_scores)
    probabilities = fitness_scores/total_fitness
    participants = np.random.choice(len(population), size=2, p=probabilities, replace=False)
    return population[participants]

def crossover1(parents):
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
    if random.random() < mutation_rate:
        idx1, idx2 = np.random.choice(len(individual), 2, replace=False)    
        if idx1 > idx2:
            idx1, idx2 = idx2 , idx1
        individual = reverse_subarray(individual,idx1,idx2)
    return individual


def reverse_subarray(arr, i, j):
    reverseArray = arr.copy()
    reverseArray[i:j + 1] = reverseArray[i:j + 1][::-1]
    return reverseArray



        
def genetic_algorithm(distances, population_size=100, generations=5000, mutation_rate=0.01, number_of_crossover=2000, elite_percent=0.1, iteration_without_improvement=200, history_of_best_results_for_generation = False, allow_dynamic_mutation = False):
    num_cities = distances.shape[0]
    population = initialize_population(population_size, num_cities)
    fitness_scores = calculate_fitness_vectorized(population, distances)
    max_mutation_rate = 0.3
    first_mutation_rate = mutation_rate

    best_index_of_first_population = np.argmin(fitness_scores)
    first_best_score = fitness_scores[best_index_of_first_population]
    number_without_improvement = 0

    current_time = datetime.datetime.now()

    formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")

    file_name = f"plik_{formatted_time}_for_{num_cities}.txt"

    formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    if history_of_best_results_for_generation == True:
        with open(file_name, 'a') as file:
            file.write("Historia najlepszych osobników dla podanych parametrów: \n")
            file.write(f"population_size = {population_size}, liczba generacji = {generations}, mutacja = {mutation_rate}, liczba krzyżówek = {number_of_crossover}, elite_percent={elite_percent}, iteracje bez poprawy = {iteration_without_improvement}  \n")



    for generation in tqdm(range(generations)):
        children = np.empty((number_of_crossover, num_cities), dtype=int)
        for i in range(0, number_of_crossover, 2):
            parents = tournament_selection(population, fitness_scores)
            offspring = crossover1(parents)
            children[i] = mutate(offspring[0], mutation_rate)
            children[i+1] = mutate(offspring[1], mutation_rate)

        children_fitness = calculate_fitness_vectorized(children, distances)
        elite_indices = np.argsort(fitness_scores)[:int(elite_percent * population_size)]
        elite_population = population[elite_indices]
        top_children_indices = np.argsort(children_fitness)[:population_size - len(elite_population)]
        population = np.vstack([elite_population, children[top_children_indices]])
        fitness_scores = calculate_fitness_vectorized(population, distances)
        best_index_of_generation = np.argmin(fitness_scores)
       
        if(generation==0):
            best_score_of_pregeneration = fitness_scores[best_index_of_generation]
            if(fitness_scores[best_index_of_generation]<first_best_score): 
                number_without_improvement = 0
            else:
                number_without_improvement += 1
                
        else:
            if(fitness_scores[best_index_of_generation]<best_score_of_pregeneration): 
                number_without_improvement = 0

                if allow_dynamic_mutation and mutation_rate < max_mutation_rate:
                    mutation_rate += 0.01
            else:
                number_without_improvement += 1 

                if number_without_improvement % 50 == 0:
                    with open(file_name, 'a') as file:
                        file.write(f"\n {sum(fitness_scores)/len(fitness_scores)}")

                if allow_dynamic_mutation and mutation_rate > first_mutation_rate:
                    mutation_rate = max(mutation_rate-0.1, first_mutation_rate )
            if(number_without_improvement == iteration_without_improvement):
                break
        best_score_of_pregeneration = fitness_scores[best_index_of_generation]
        if history_of_best_results_for_generation == True and number_without_improvement == 0:
            average_route = sum(fitness_scores)
            save_the_best_individual(population[best_index_of_generation], best_score_of_pregeneration, generation ,file_name, average_route )
        


    best_index = np.argmin(fitness_scores)
    saveData(population[best_index],fitness_scores[best_index],population_size,generations,number_of_crossover,mutation_rate,iteration_without_improvement)
    return population[best_index], fitness_scores[best_index]












#readData=pd.read_csv("Dane_TSP_127.csv",sep=";", decimal=',')
#readData=pd.read_csv("Dane_TSP_76.csv",sep=";", decimal=',')
readData=pd.read_csv("Dane_TSP_48.csv",sep=";", decimal=',')

#readData=pd.read_csv("Miasta29.csv",sep=";", decimal=',')

distance_matrix = readData.iloc[:,1:].astype(float).to_numpy()



for i in range(6):
    best_route, best_distance = genetic_algorithm(distance_matrix,50,generations=650,mutation_rate=0.05, number_of_crossover = 1000, elite_percent=0.1, iteration_without_improvement=27000)

print("Best Route:", best_route)
print("Best Distance:", best_distance)