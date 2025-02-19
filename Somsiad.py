import numpy as np
import pandas as pd


def nearest_neighbor_tsp(distance_matrix, start_index):
    num_cities = distance_matrix.shape[0]
    visited_cities = [start_index]
    total_distance = 0

    distance_matrix = distance_matrix.astype(float)
    all_indices = set(range(num_cities))
    while len(visited_cities) < num_cities:
        remaining_indices = list(all_indices - set(visited_cities))
        next_index = np.argmin(distance_matrix[visited_cities[-1], remaining_indices])
        next_city = remaining_indices[next_index]
        total_distance += distance_matrix[visited_cities[-1], next_city]
        visited_cities.append(next_city)
    total_distance += distance_matrix[visited_cities[-1], visited_cities[0]]
    return total_distance, visited_cities

def find_minimum_route(distance_matrix):
    list_of_results = []
    for i in range(distance_matrix.shape[0]):
        list_of_results.append(nearest_neighbor_tsp(distance_matrix, i))
    return list(filter(lambda x: x[0] == min_value_in_list(list_of_results), list_of_results))

def min_value_in_list(list_of_results):
    return min(list_of_results, key=lambda x: x[0])[0]


if __name__ == "__main__":
    # read_data = pd.read_csv("Dane_TSP_127.csv", sep=";")
    read_data = pd.read_csv("Dane_TSP_48.csv", sep=";", decimal=",")
    # read_data = pd.read_csv("Dane_TSP_76.csv", sep=";", decimal=",")
    distance_matrix = read_data.iloc[:, 1:].astype(float).to_numpy()
    print(find_minimum_route(distance_matrix))