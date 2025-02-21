import numpy as np
import pandas as pd
import random
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


def save_data(best_route, distance, method, num_iterations, filename=get_result_file_path(f"Wspinaczka_records48.txt")):
    with open(filename, 'a') as resultFile:
        resultFile.write("\n" + "=" * 25 + "\n")
        for element in best_route:
            resultFile.write(str(element + 1) + ' ')
        resultFile.write(str(distance))
        resultFile.write("\n" + "Metoda: " + str(method))
        resultFile.write("\n" + "Liczba iteracji bez poprawy: " + str(num_iterations))


def get_random_route_cities(number_of_cities):
    return random.sample(range(0, number_of_cities), number_of_cities)


def swap_cities(city_order, first_index, second_index):
    first_city = city_order[first_index]
    city_order[first_index] = city_order[second_index]
    city_order[second_index] = first_city
    return city_order


def get_cities(city_order: np.array, first_index: int, second_index: int):
    def city(index):
        return city_order[index % len(city_order)]
    neighbour_of_first_index = np.array([city(first_index - 1), city(first_index + 1)])
    neighbour_of_second_index = np.array([city(second_index - 1), city(second_index + 1)])
    return neighbour_of_first_index, neighbour_of_second_index


def check_if_we_get_better_route(distance_matrix: np.array, city_order: np.array, list_of_neighbours, first_index_swapping, second_index_swapping):
    if if_swap_neighbour(len(city_order) - 1, first_index_swapping, second_index_swapping) == False:
        previous_route = check_route_with_neighbour(distance_matrix, city_order, first_index_swapping, list_of_neighbours[0]) + check_route_with_neighbour(distance_matrix, city_order, second_index_swapping, list_of_neighbours[1])
        new_route = check_route_with_neighbour(distance_matrix, city_order, first_index_swapping, list_of_neighbours[1]) + check_route_with_neighbour(distance_matrix, city_order, second_index_swapping, list_of_neighbours[0])
        if new_route < previous_route:
            return (swap_cities(city_order, first_index_swapping, second_index_swapping), True)
        return (city_order, False)
    first_idx = first_index_swapping
    second_idx = second_index_swapping
    if second_index_swapping < first_index_swapping:
        first_idx = second_index_swapping
        second_idx = first_index_swapping
    distance_change = calculate_route_change_for_neighbour(distance_matrix, city_order, first_idx, second_idx)
    if distance_change < 0:
        return (swap_cities(city_order, first_index_swapping, second_index_swapping), True)
    return (city_order, False)


def if_swap_neighbour(length, first_index_swapping, second_index_swapping):
    return abs(second_index_swapping - first_index_swapping) <= 1 or (first_index_swapping == 0 and second_index_swapping == length) or (second_index_swapping == 0 and first_index_swapping == length)


def reverse_subarray(array, start_index, end_index):
    reversed_array = array.copy()
    reversed_array[start_index:end_index + 1] = reversed_array[start_index:end_index + 1][::-1]
    return reversed_array


def insert_at_index(order, index, insertion_index):
    element = order[index]
    new_order = np.delete(order, index)
    new_order = np.insert(new_order, insertion_index, element)
    return new_order


def check_route_with_neighbour(distance_matrix, city_order, index, neighbour_cities):
    return distance_matrix[city_order[index], neighbour_cities[0]] + distance_matrix[city_order[index], neighbour_cities[1]]


def calculate_route_change_for_neighbour(distance_matrix, city_order, first_index, second_index):
    length = len(city_order)
    if first_index == 0 and second_index == length - 1:
        length_before = (distance_matrix[city_order[0], city_order[1]] +
                         distance_matrix[city_order[-1], city_order[-2]])
        length_after = (distance_matrix[city_order[0], city_order[-2]] +
                        distance_matrix[city_order[-1], city_order[1]])
        return length_after - length_before

    length_before = (distance_matrix[city_order[first_index - 1], city_order[first_index]] +
                     distance_matrix[city_order[first_index + 1], city_order[(first_index + 2) % length]])

    length_after = (distance_matrix[city_order[first_index - 1], city_order[first_index + 1]] +
                    distance_matrix[city_order[first_index], city_order[(first_index + 2) % length]])

    return length_after - length_before


def calculate_route_change_for_insertion(distance_matrix: np.array, city_order: np.array, index_of_city_in_city_order, index_to_insert):
    length_of_city_order = len(city_order)
    if if_swap_neighbour(length_of_city_order - 1, index_of_city_in_city_order, index_to_insert) == False:
        if index_of_city_in_city_order < index_to_insert:
            length_before = distance_matrix[city_order[index_of_city_in_city_order], city_order[index_of_city_in_city_order - 1]] + distance_matrix[city_order[index_of_city_in_city_order], city_order[(index_of_city_in_city_order + 1) % length_of_city_order]] + distance_matrix[city_order[index_to_insert], city_order[(index_to_insert + 1) % length_of_city_order]]

            length_after = distance_matrix[city_order[index_of_city_in_city_order], city_order[index_to_insert]] + distance_matrix[city_order[index_of_city_in_city_order], city_order[(index_to_insert + 1) % length_of_city_order]] + distance_matrix[city_order[index_of_city_in_city_order + 1], city_order[(index_of_city_in_city_order - 1)]]
            return length_after - length_before
        length_before = (distance_matrix[city_order[index_of_city_in_city_order], city_order[index_of_city_in_city_order - 1]] +
                         distance_matrix[city_order[index_of_city_in_city_order], city_order[(index_of_city_in_city_order + 1) % length_of_city_order]] +
                         distance_matrix[city_order[index_to_insert], city_order[index_to_insert - 1]])
        length_after = (distance_matrix[city_order[index_of_city_in_city_order], city_order[index_to_insert]] +
                        distance_matrix[city_order[index_of_city_in_city_order], city_order[index_to_insert - 1]] +
                        distance_matrix[city_order[(index_of_city_in_city_order + 1) % length_of_city_order], city_order[(index_of_city_in_city_order - 1)]])
        return length_after - length_before
    elif abs(index_of_city_in_city_order - index_to_insert) == 1:
        first_idx = index_of_city_in_city_order
        second_idx = index_to_insert
        if index_to_insert < index_of_city_in_city_order:
            first_idx = index_to_insert
            second_idx = index_of_city_in_city_order
        return calculate_route_change_for_neighbour(distance_matrix, city_order, first_idx, second_idx)
    return 0


def calculate_route_change_for_reverse(distance_matrix: np.array, city_order: np.array, first_index, second_index):
    length_of_city_order = len(city_order)
    if first_index == 0 and second_index == length_of_city_order - 1:
        return 0
    if first_index - second_index == -1:
        return calculate_route_change_for_neighbour(distance_matrix, city_order, first_index, second_index)
    return (distance_matrix[city_order[second_index], city_order[first_index - 1]] +
            distance_matrix[city_order[first_index], city_order[(second_index + 1) % length_of_city_order]]) - (
           distance_matrix[city_order[first_index], city_order[first_index - 1]] +
           distance_matrix[city_order[second_index], city_order[(second_index + 1) % length_of_city_order]])


def check_if_we_get_better_route_for_reverse(distance_matrix: np.array, city_order: np.array, first_index, second_index):
    distance_change = calculate_route_change_for_reverse(distance_matrix, city_order, first_index, second_index)
    if distance_change < 0:
        return reverse_subarray(city_order, first_index, second_index), True
    return city_order, False


def check_if_we_get_better_route_for_insertion(distance_matrix: np.array, city_order: np.array, first_index, second_index):
    distance_change = calculate_route_change_for_insertion(distance_matrix, city_order, first_index, second_index)
    if distance_change < 0:
        return insert_at_index(city_order, first_index, second_index), True
    return city_order, False


def get_sum_of_cities(distance_matrix, city_orders):
    total_distance = 0
    for i in range(-1, len(city_orders) - 1):
        total_distance += distance_matrix[city_orders[i], city_orders[i + 1]]
    return total_distance


def climbing_algorithm(distance_matrix, city_order, max_iterations_without_improvement, cities, method="reverse", max_iterations=None):
    iteration_count = 0
    total_iterations = 0
    max_iterations = max_iterations_without_improvement if max_iterations is None else max_iterations
    while iteration_count < max_iterations:
        indices = random.sample(cities, 2)
        new_city_order = None
        if method == "swapping":
            neighbours = get_cities(city_order, indices[0], indices[1])
            new_city_order = check_if_we_get_better_route(distance_matrix, city_order, neighbours, indices[0], indices[1])
        elif method == "reverse":
            if indices[0] > indices[1]:
                indices[0], indices[1] = indices[1], indices[0]
            new_city_order = check_if_we_get_better_route_for_reverse(distance_matrix, city_order, indices[0], indices[1])
        else:
            if indices[0] > indices[1]:
                indices[0], indices[1] = indices[1], indices[0]
            new_city_order = check_if_we_get_better_route_for_insertion(distance_matrix, city_order, indices[0], indices[1])
        city_order = new_city_order[0]
        iteration_count += 1
        total_iterations += 1
        if new_city_order[1] and max_iterations is None:
            iteration_count = 0
    return city_order


def make_iteration(repetitions, distance_matrix, max_iterations_without_improvement, acceptance_value, method="reverse", max_iterations=None):
    for _ in range(repetitions):
        city_order = get_random_route_cities(distance_matrix.shape[0])
        cities = range(len(city_order))
        final_city_order = climbing_algorithm(distance_matrix, city_order, max_iterations_without_improvement, cities, method, max_iterations)
        final_distance = get_sum_of_cities(distance_matrix, final_city_order)
        print(final_distance)
        print(final_city_order)

        if final_distance < acceptance_value:
            save_data(final_city_order, final_distance, method, max_iterations_without_improvement)


if __name__ == "__main__":
    # readData=pd.read_csv("Miasta29.csv",sep=";", decimal=",")
    readData = pd.read_csv("Dane_TSP_48.csv", sep=";", decimal=",")
    readData = pd.read_csv("Dane_TSP_76.csv", sep=";", decimal=",")
    readData = pd.read_csv("Dane_TSP_127.csv", sep=";", decimal=",")
    distance_matrix = readData.iloc[:, 1:].astype(float).to_numpy()
    make_iteration(1, distance_matrix, 2200, 2000, method="reverse", max_iterations=1000000)