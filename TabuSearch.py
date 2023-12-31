import pandas as pd
import random
import copy
from tqdm import tqdm

# Funkcja obliczająca odległość między dwoma miastami
def calculate_distance(route, cities_df):
    distance = 0
    for i in range(len(route) - 1):
        city1 = route[i]
        city2 = route[i + 1]
        distance += calculate_euclidean_distance(city1, city2, cities_df)
    distance += calculate_euclidean_distance(route[-1], route[0], cities_df)  # Return to the starting city
    return distance

# Funkcja obliczająca odległość euklidesową między dwoma miastami
def calculate_euclidean_distance(city1, city2, cities_df):
    dist = cities_df.loc[city1, city2]
    return dist

# Generowanie sąsiedztwa poprzez zamianę dwóch miast w trasie
def generate_neighborhood(route):
    neighborhood = []
    for i in range(len(route)):
        for j in range(i + 1, len(route)):
            neighbor = copy.deepcopy(route)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            neighborhood.append(neighbor)
    return neighborhood

# Wybieranie najlepszego sąsiedztwa spośród sąsiadów, które nie są na liście tabu
def get_best_neighborhood(neighborhood, tabu_list, cities_df):
    best_distance = float('inf')
    best_neighbor = None
    for neighbor in neighborhood:
        distance = calculate_distance(neighbor, cities_df)
        if distance < best_distance and neighbor not in tabu_list:
            best_distance = distance
            best_neighbor = neighbor
    return best_neighbor

# Aktualizacja listy tabu
def update_tabu_list(tabu_list, new_solution, tabu_tenure):
    tabu_list.append(new_solution)
    if len(tabu_list) > tabu_tenure:
        tabu_list.pop(0)

def tabu_search(cities_df, output_file, iterations, tabu_tenure=7):

    num_cities = len(cities_df)
    tabu_list = []
    overall_best_solution = list(range(1, num_cities + 1))
    random.shuffle(overall_best_solution)

    for n in tqdm(range(iterations)):
        current_solution = list(range(1, num_cities + 1))
        random.shuffle(current_solution)
        best_solution = current_solution
        rep_counter = 0
        # i=0
        while True:

            neighborhood = generate_neighborhood(best_solution)
            best_neighbor = get_best_neighborhood(neighborhood, tabu_list, cities_df)

            if best_neighbor is not None:
                current_solution = best_neighbor
                current_distance = calculate_distance(current_solution, cities_df)
                best_distance = calculate_distance(best_solution, cities_df)

                if current_distance < best_distance:
                    best_solution = current_solution
                    rep_counter = 0
                else:
                    rep_counter += 1
                update_tabu_list(tabu_list, best_neighbor, tabu_tenure)

            if rep_counter == 5:
                break

        if(calculate_distance(best_solution, cities_df) < calculate_distance(overall_best_solution, cities_df)):
            overall_best_solution = best_solution

    # Zapisz wynik do pliku txt
    with open(output_file, "a") as file:
        file.write(f"----------------\nIterations: {iterations}\nBest solution: {overall_best_solution}\nDistance: {calculate_distance(overall_best_solution, cities_df)}")
    return overall_best_solution

# Przykładowe użycie
excel_file = 'Dane_TSP_127.xlsx'
cities_df = pd.read_excel(excel_file, index_col=0)
best_solution = tabu_search(cities_df, "ResultsTabuSearch.txt", iterations=35, tabu_tenure=10)

print("Best solution:", best_solution)
print("Best distance:", calculate_distance(best_solution, cities_df))