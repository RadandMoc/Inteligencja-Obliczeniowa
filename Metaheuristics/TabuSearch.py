import sys
import numpy as np
import pandas as pd
import random
import copy
import time
from tqdm import tqdm
from enum import Enum
import os


def create_results_dir():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    return results_dir

def get_result_file_path(filename):
    results_dir = create_results_dir()
    return os.path.join(results_dir, filename)

class Method(Enum):
    Swap = 1
    Insertion = 2
    Reverse = 3

def get_difference(method, new_route, current_route, distance_matrix, current_distance, index1, index2):
    if method == Method.Swap:
        neighbours = get_cities(new_route, index1, index2)
        new_distance = (current_distance +
                        check_if_we_get_better_route_swapping(distance_matrix, current_route, neighbours, index1, index2))
        new_route = swap_cities(new_route, index1, index2)
        return new_distance, new_route
    elif method == Method.Insertion:
        new_distance = (current_distance +
                        check_if_we_get_better_route_insertion(distance_matrix, current_route, index1, index2))
        new_route = insert_method(new_route, index1, index2)
        return new_distance, new_route
    else:
        if index1 > index2:
            index1, index2 = index2, index1
        new_distance = (current_distance +
                        check_if_we_get_better_route_reverse(distance_matrix, current_route, index1, index2))
        new_route = reverse_method(new_route, index1, index2)
        return new_distance, new_route

def get_cities(city_order: np.array, first_index: int, second_index: int):
    def city(index):
        return city_order[index % len(city_order)]

    neighbour_of_first_index = np.array([city(first_index - 1), city(first_index + 1)])
    neighbour_of_second_index = np.array([city(second_index - 1), city(second_index + 1)])
    return neighbour_of_first_index, neighbour_of_second_index

def calculate_route_distance(route, cities_df):
    total_distance = 0
    for i in range(-1, len(route) - 1):
        total_distance += cities_df[route[i], route[i + 1]]
    return total_distance

def swap_cities(city_order, first_index, second_index):
    first_city = city_order[first_index]
    city_order[first_index] = city_order[second_index]
    city_order[second_index] = first_city
    return city_order

def check_if_we_get_better_route_swapping(distance_matrix: np.array, city_order: np.array, list_of_neighbours, first_index_swapping, second_index_swapping):
    if if_swap_neighbour(len(city_order) - 1, first_index_swapping, second_index_swapping) == False:
        previous_route = check_route_with_neighbour(distance_matrix, city_order, first_index_swapping, list_of_neighbours[0]) + check_route_with_neighbour(distance_matrix, city_order, second_index_swapping, list_of_neighbours[1])
        new_route = check_route_with_neighbour(distance_matrix, city_order, first_index_swapping, list_of_neighbours[1]) + check_route_with_neighbour(distance_matrix, city_order, second_index_swapping, list_of_neighbours[0])
        return new_route - previous_route
    first_idx = first_index_swapping
    second_idx = second_index_swapping
    if second_index_swapping < first_index_swapping:
        first_idx = second_index_swapping
        second_idx = first_index_swapping
    distance_change = calculate_route_change_for_neighbour(distance_matrix, city_order, first_idx, second_idx)
    return distance_change

def check_if_we_get_better_route_insertion(distance_matrix: np.array, city_order: np.array, index_of_city_in_city_order, index_to_insert):
    length_of_city_order = len(city_order)
    if if_swap_neighbour(length_of_city_order - 1, index_of_city_in_city_order, index_to_insert) == False:
        if index_of_city_in_city_order < index_to_insert:
            length_before = (distance_matrix[city_order[index_of_city_in_city_order], city_order[index_of_city_in_city_order - 1]]
                            + distance_matrix[city_order[index_of_city_in_city_order], city_order[(index_of_city_in_city_order + 1) % length_of_city_order]]
                            + distance_matrix[city_order[index_to_insert], city_order[(index_to_insert + 1) % length_of_city_order]])
            length_after = (distance_matrix[city_order[index_of_city_in_city_order], city_order[index_to_insert]]
                           + distance_matrix[city_order[index_of_city_in_city_order], city_order[(index_to_insert + 1) % length_of_city_order]]
                           + distance_matrix[city_order[index_of_city_in_city_order + 1], city_order[(index_of_city_in_city_order - 1)]])
            return length_after - length_before
        length_before = (distance_matrix[city_order[index_of_city_in_city_order], city_order[index_of_city_in_city_order - 1]]
                        + distance_matrix[city_order[index_of_city_in_city_order], city_order[(index_of_city_in_city_order + 1) % length_of_city_order]]
                        + distance_matrix[city_order[index_to_insert], city_order[index_to_insert - 1]])
        length_after = (distance_matrix[city_order[index_of_city_in_city_order], city_order[index_to_insert]]
                       + distance_matrix[city_order[index_of_city_in_city_order], city_order[index_to_insert - 1]]
                       + distance_matrix[city_order[(index_of_city_in_city_order + 1) % length_of_city_order], city_order[(index_of_city_in_city_order - 1)]])
        return length_after - length_before
    elif abs(index_of_city_in_city_order - index_to_insert) == 1:
        first_idx = index_of_city_in_city_order
        second_idx = index_to_insert
        if index_to_insert < index_of_city_in_city_order:
            first_idx = index_to_insert
            second_idx = index_of_city_in_city_order
        return calculate_route_change_for_neighbour(distance_matrix, city_order, first_idx, second_idx)
    return 0

def insert_method(order, index, insertion_index):
    element = order[index]
    new_order = np.delete(order, index)
    new_order = np.insert(new_order, insertion_index, element)
    return new_order

def check_if_we_get_better_route_reverse(distance_matrix: np.array, city_order: np.array, first_index, second_index):
    length_of_city_order = len(city_order)
    if first_index == 0 and second_index == length_of_city_order - 1:
        return 0
    if first_index - second_index == -1:
        return calculate_route_change_for_neighbour(distance_matrix, city_order, first_index, second_index)
    length_before = (distance_matrix[city_order[first_index], city_order[first_index - 1]]
                    + distance_matrix[city_order[second_index], city_order[(second_index + 1) % length_of_city_order]])
    length_after = (distance_matrix[city_order[second_index], city_order[first_index - 1]]
                   + distance_matrix[city_order[first_index], city_order[(second_index + 1) % length_of_city_order]])
    return length_after - length_before

def reverse_method(order, first_index, second_index):
    reverse_order = order.copy()
    reverse_order[first_index:second_index + 1] = reverse_order[first_index:second_index + 1][::-1]
    return reverse_order

def if_swap_neighbour(length, first_index_swapping, second_index_swapping):
    return abs(second_index_swapping - first_index_swapping) <= 1 or (
                first_index_swapping == 0 and second_index_swapping == length) or (
                second_index_swapping == 0 and first_index_swapping == length)

def check_route_with_neighbour(distance_matrix, city_order, index, neighbour_cities):
    return distance_matrix[city_order[index], neighbour_cities[0]] + distance_matrix[city_order[index], neighbour_cities[1]]

def calculate_route_change_for_neighbour(distance_matrix, city_order, index1, index2):
    length = len(city_order)
    if index1 == 0 and index2 == length - 1:
        length_before = (distance_matrix[city_order[0], city_order[1]]
                         + distance_matrix[city_order[-1], city_order[-2]])
        length_after = (distance_matrix[city_order[0], city_order[-2]]
                        + distance_matrix[city_order[-1], city_order[1]])
        return length_after - length_before

    length_before = (distance_matrix[city_order[index1 - 1], city_order[index1]]
                     + distance_matrix[city_order[index1 + 1], city_order[(index1 + 2) % length]])

    length_after = (distance_matrix[city_order[index1 - 1], city_order[index1 + 1]]
                    + distance_matrix[city_order[index1], city_order[(index1 + 2) % length]])

    return length_after - length_before

def calculate_euclidean_distance(city1, city2, cities_df):
    dist = cities_df[city1, city2]
    return dist

def generate_neighborhood(route):
    neighborhood = []
    moves = []
    neighborhood_moves = []
    for i in range(len(route)):
        for j in range(i + 1, len(route)):
            neighbor = copy.deepcopy(route)
            move = [i, j]
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            neighborhood.append(neighbor)
            moves.append(move)
    neighborhood_moves = list(zip(neighborhood, moves))
    return neighborhood_moves

def calculate_distance(neighborhood_moves, current_route, current_distance, cities_df, method):
    route = neighborhood_moves[0]
    index1 = neighborhood_moves[1][0]
    index2 = neighborhood_moves[1][1]
    city1 = route[index1]
    city2 = route[index2]
    new_route = np.copy(current_route)
    new_distance, new_route = get_difference(method, new_route, current_route, cities_df, current_distance, index1, index2)
    return new_distance, new_route, index1, index2

def update_tabu_list(tabu_list, new_solution, tabu_tenure):
    tabu_list.append(new_solution)
    if len(tabu_list) > tabu_tenure:
        tabu_list.pop(0)

def tabu_search(csv_file, output_file, iterations, iterations_without_improvement, method, tabu_tenure=7):
    start_time = time.time()
    read_data = pd.read_csv(csv_file, sep=";", decimal=',')
    cities_df = read_data.iloc[:, 1:].astype(float).to_numpy()

    num_cities = len(cities_df)
    current_solution = list(range(0, num_cities))
    random.shuffle(current_solution)
    overall_best_solution = current_solution
    overall_best_distance = calculate_route_distance(overall_best_solution, cities_df)

    tabu_list = []
    no_improvement_counter = 0
    for n in tqdm(range(iterations)):
        neighbor_current_distance = sys.maxsize
        best_distance_in_tabu = sys.maxsize
        best_solution_in_tabu = []
        best_solution = current_solution
        best_distance = calculate_route_distance(current_solution, cities_df)
        neighborhood_moves = generate_neighborhood(best_solution)
        best_move = []
        best_neighbor = None
        best_neighbor_distance = 0

        for neighbor in neighborhood_moves:
            new_distance, new_route, index1, index2 = calculate_distance(neighbor, best_solution, best_distance, cities_df, method)
            if index1 > index2:
                index1, index2 = index2, index1
            neighbor_move = [index1, index2]

            if neighbor_move in tabu_list:
                if new_distance < best_distance_in_tabu:
                    best_distance_in_tabu = new_distance
                    best_solution_in_tabu = new_route
                continue
            if new_distance < neighbor_current_distance:
                neighbor_current_distance = new_distance
                best_neighbor = new_route
                best_neighbor_distance = new_distance
                best_move = neighbor_move

        if best_neighbor is not None and best_move is not None:
            current_solution = best_neighbor
            current_distance = best_neighbor_distance
            if current_distance < best_distance:
                best_solution = current_solution
                best_distance = current_distance
                update_tabu_list(tabu_list, best_move, tabu_tenure)
            elif best_distance_in_tabu < overall_best_distance:
                best_solution = best_solution_in_tabu
                best_distance = best_distance_in_tabu
                update_tabu_list(tabu_list, [], tabu_tenure)
            else:
                best_solution = current_solution
                best_distance = current_distance
                update_tabu_list(tabu_list, best_move, tabu_tenure)

        if best_distance < overall_best_distance:
            overall_best_distance = best_distance
            overall_best_solution = best_solution
            no_improvement_counter = 0
        else:
            no_improvement_counter += 1
            if no_improvement_counter >= iterations_without_improvement:
                break

    end_time = time.time()
    elapsed_time = end_time - start_time
    with open(output_file, "a") as file:
        file.write(f"----------------\nFile: {csv_file}\nIterations: {iterations}\nIterations without improvement: {iterations_without_improvement}\nMethod: {method}\nTabu tenure: {tabu_tenure}\nBest solution: {overall_best_solution}\nDistance: {calculate_route_distance(overall_best_solution, cities_df)}\nTime: {elapsed_time}\n")
        print(f"----------------\nFile: {csv_file}\nIterations: {iterations}\nIterations without improvement: {iterations_without_improvement}\nMethod: {method}\nTabu tenure: {tabu_tenure}\nBest solution: {overall_best_solution}\nDistance: {calculate_route_distance(overall_best_solution, cities_df)}\nTime: {elapsed_time}\n")
    return overall_best_solution


if __name__ == "__main__":
    # Przykładowe użycie
    best_solution = tabu_search('Dane_TSP_76.csv', get_result_file_path("ResultsTabuSearch.txt"), iterations=10000, iterations_without_improvement=500, method=Method.Reverse, tabu_tenure=400)