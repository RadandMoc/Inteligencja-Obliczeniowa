import numpy as np
import pandas as pd
import random
import math
import time
import sys
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

# FUNKCJA DO ZAPISU DO PLIKU
def save_data(best_overall, initial_temperature, cooling_rate, num_iterations, min_temp, temp_red, metoda, filename):
        with open(filename, 'a') as resultFile:
            resultFile.write("\n" + "=" * 25 + "\n")
            for element in best_overall[1]:
                resultFile.write(str(element+1) + ' ')
            resultFile.write("\n" + "Najlepsza odleglosc: " + str(best_overall[0]))
            resultFile.write("\n" + "Poczatkowa temparatura: " + str(initial_temperature))
            resultFile.write("\n" + "Wsp. chlodzenia: " + str(cooling_rate))
            resultFile.write("\n" + "Liczba iteracji dla jednej temperatury: " + str(num_iterations))
            resultFile.write("\n" + "Minimalna temperatura: " + str(min_temp))
            resultFile.write("\n" + "Metoda redukcji temperatury: " + temp_red)
            resultFile.write("\n" + "Metoda: " + metoda)

# FUNKCJA LOSUJĄCA LOSOWĄ TRASE
def get_random_route_cities(numberOfCities):
    return random.sample(range(0, numberOfCities), numberOfCities)

# FUNKCJA LICZĄCA CAŁKOWITĄ SUMĘ TRAS, PRZYDATNA TYLKO RAZ, GDYŻ POTEM ODLEGŁOŚĆ LICZYMY NA PODSTAWIE RÓŻNIC
def get_sum_of_cities(cityOrders, distanceMatrix):
    sum = 0
    for i in range(-1,len(cityOrders)-1):
        sum = sum + distanceMatrix[cityOrders[i], cityOrders[i+1]]
    return sum

# FUNKCJA DO SWAPINGU
def swap_cities(cityOrder,firstIdx,Secondidx):
    firstCity = cityOrder[firstIdx]
    cityOrder[firstIdx] = cityOrder[Secondidx]
    cityOrder[Secondidx] = firstCity
    return cityOrder

def get_difference(meth, new_r, current_r, distance_m, current_dist, idx1, idx2):
    if (meth == "swap"):
        neighbours = get_cities(new_r, idx1, idx2)  # Biorę sąsiadów indeksów
        new_distance = (current_dist +
                        check_If_we_get_better_route_swapping(distance_m, current_r, neighbours, idx1,
                                                        idx2))  # Liczę dystans gdybym zamienił miasta
        new_route = swap_cities(new_r, idx1, idx2)  # Nowa trasa po swappingu
        return new_distance, new_route
    elif (meth == "insertion"):
        new_distance = (current_dist +
                        check_if_we_get_better_route_for_insertion(distance_m, current_r, idx1,
                                                            idx2))  # Liczę dystans gdybym zmienił kolejność miast
        new_route = insert_method(new_r, idx1, idx2)  # Nowa trasa po insercji
        return new_distance, new_route
    else:
        if (idx1 > idx2):
            idx1, idx2 = idx2, idx1  # Zamieniam indeksy, aby spełnić warunek funkcji checkIfGetBetter...
        new_distance = (current_dist +
                        check_if_we_get_better_route_for_reverse(distance_m, current_r, idx1, idx2))
        new_route = reverse_method(new_r, idx1, idx2)
        return new_distance, new_route

# FUNKCJA LICZĄCA PRAWDOPODOBIEŃSTWO ZAMIANY
def acceptance_probability(old_distance, new_distance, temperature):
    if new_distance < old_distance:
        return 1.0
    return math.exp((old_distance - new_distance) / temperature)

# POJEDYNCZA ITERACJA SYMULOWANEGO WYŻARZANIA
def simulated_annealing(distance_matrix, temperature, cooling_rate, num_iterations, min_temp, temp_red, method="swap"):
    current_route = get_random_route_cities(len(distance_matrix)) # Początkowa trasa - randomowa
    current_distance = get_sum_of_cities(current_route, distance_matrix) # Dystans początkowej trasy
    best_one_iteration = current_distance # Globalne najlepsze rozwiązanie w jednej iteracji
    best_one_route = np.copy(current_route)
    new_distance = 0

    if(temp_red == 'slow'):
        num_iterations = 1

    while(temperature>min_temp): # Pętla się wykonuje, aż temperatura nie będzie miała pożądanej temperatury
        for iteration in range(num_iterations): # Pętla wykonuje się określoną liczbę iteracji
            new_route = np.copy(current_route)
            idx1, idx2 = random.sample(range(len(new_route)), 2) # Losujemy 2 randomowe indexy od 0 do (długości trasy-1)

            new_distance, new_route = get_difference(method, new_route, current_route, distance_matrix, current_distance, idx1, idx2)

            if acceptance_probability(current_distance, new_distance, temperature) > random.random(): # Jeśli prawdopodobieństwo wybrania trasy jest większe niż losowa liczba z zakresu 0 do 1, to:
                current_route = np.copy(new_route) # Zamieniam aktualną trasą na tą nową wygenenerowaną ostatnio
                current_distance = new_distance # Zmieniam aktualny dystans na ten wygenerowany ostatnio
                if(current_distance < best_one_iteration):
                    best_one_iteration = current_distance
                    best_one_route = current_route
        if(temp_red == 'slow'):
            temperature = temperature/(1+temperature*cooling_rate)
        else:
            temperature *= (1 - cooling_rate) # Schładzam temperaturę po wykonaniu określonej liczby iteracji

    return current_route, current_distance, [best_one_iteration, best_one_route] # Zwracam trasę końcową wraz z jej dystansem oraz najlepszą możliwą trasę którą udało się wygenerować i przypadkowo z tego optimum lokalnego wyszliśmy. Dodatkowo zwracam najlepsze globalne rozwiązanie z jednej iteracji.

def run_simulated_annealing_multiple_times(distance_matrix, num_runs,
                                       initial_temperature, cooling_rate, num_iterations, min_temp, acc_value,
                                       filename, temp_red, method="swap"): # Funkcja wykonującaco symulowane wyżarzanie num_runs

    if(method != "swap" and method != "reverse" and method != "insertion"): # Sprawdzanie czy wprowadziliśmy istniejącą metodę zamiany
        print("Method doesn't exist.")
        return [0, 0, [0, 0]]

    best_finished = [sys.maxsize, []] # Najlepsza odległość pod koniec z num_runs razy wykonanego algorytmu wraz z trasą
    best_overall = [sys.maxsize, []] # Najlepsza odległość ogólnie z num_rns razy wykonanego algorytmu wraz z trasą (może być pod koniec, może być nie pod koniec)

    for i in range(num_runs):
        data_run = simulated_annealing(distance_matrix, initial_temperature, cooling_rate, num_iterations, min_temp, temp_red, method) # Wykonanie algorytmu symulowanego wyżarzania
        best_route, best_distance, glbestone = data_run[0], data_run[1], data_run[2] # Kolejno najlepsza odległość końcowa, najlepszy dystans końcowy , najlepsze rozwiązanie w całym okresie wykonywanego algorytmu, najlepsze rozwiązanie w okresie jednej iteracji
        print("Odległość na końcu:", best_distance) # Printuje nam końcowy dystans

        if best_distance < best_finished[0]: # Jeśli końcowy dystans jest lepszy od najlepszego końcowego dystansu, to:
            best_finished = [best_distance, best_route] # Przypisujemy do zmiennej best_finished dystans i trasę


        if (best_overall[0] > glbestone[0]): # Jeśli najlepsze rozwiązanie ogólne z iteracji algorytmu symulowanego wyżarzania jest najlepsze, (np. gdy rozwiązanie z 4 iteracji wykonania całego algorytmu jest lepsze niż rozwiązanie z 2 iteracji, które było do tego czasu najlepszym):
            best_overall = glbestone # Przypisuję do zmiennej best_overall listę posiadającą trasę wraz z odległością

        if (glbestone[0] < acc_value): # Jeśli najlepsza globalna odległość jest lepsza od tej, jaką ustalimy, to zapisujemy do pliku
            save_data(glbestone, initial_temperature=initial_temperature,
                     cooling_rate=cooling_rate, num_iterations=num_iterations,
                     min_temp=min_temp, temp_red=temp_red, metoda=method, filename=filename)

    return best_finished[0], best_finished[1], best_overall # Zwracamy najlepszą trasę końcową ze wszystkich wraz z trasą oraz zwracamy najlepsze globalne rozwiązanie

def if_swap_neighbour(lenght, firstIndexSwapping, SecondIndexSwapping): # Sprawdzamy czy indeksy są sąsiadami
    return abs(SecondIndexSwapping - firstIndexSwapping) <= 1 or (
                firstIndexSwapping == 0 and SecondIndexSwapping == lenght) or (
                SecondIndexSwapping == 0 and firstIndexSwapping == lenght)

def check_route_with_neighbour(distanceMatrix, cityOrder, index, neighbourCities): # Liczy nam trasę indeksu z dwoma sąsiadami
    return distanceMatrix[cityOrder[index], neighbourCities[0]] + distanceMatrix[cityOrder[index], neighbourCities[1]]

def get_cities(cityOrder: np.array, firstIdx: int, secondIdx: int): # Funkcja, której zadaniem jest zebranie sąsiadów dla dwóch indeksów
    def city(index):
        return cityOrder[index % len(cityOrder)]

    neighbourOfFirstIndex = np.array([city(firstIdx - 1), city(
        firstIdx + 1)])  # Zbieram sąsiadów dla pierwszego miasta w funkcji city jest modulo by uniknac błedu wyjscia indeksu poza zakres dla firstidx=len(cityOrder)-1
    neighbourOfSecondIndex = np.array([city(secondIdx - 1), city(secondIdx + 1)])
    return neighbourOfFirstIndex, neighbourOfSecondIndex # Zwracamy dwóch sąsiadów dla dwóch indeksów (nr miast)

def check_If_we_get_better_route_swapping(distanceMatrix: np.array, cityOrder: np.array, listOfNeighbour, firstIndexSwapping,
                            SecondIndexSwapping): # Funkcja sprawdzająca różnice odległości gdyby dokonał się swapping
    if if_swap_neighbour(len(cityOrder) - 1, firstIndexSwapping, SecondIndexSwapping) == False: # Sprawdzamy czy zamieniamy sąsiadów
        previousRoute = check_route_with_neighbour(distanceMatrix, cityOrder, firstIndexSwapping,
                                                listOfNeighbour[0]) + check_route_with_neighbour(distanceMatrix, cityOrder,
                                                                                              SecondIndexSwapping,
                                                                                              listOfNeighbour[1]) # Odległości pomiędzy sąsiadami indeksów, których nie zamieniamy jeszcze
        newRoute = check_route_with_neighbour(distanceMatrix, cityOrder, firstIndexSwapping,
                                           listOfNeighbour[1]) + check_route_with_neighbour(distanceMatrix, cityOrder,
                                                                                         SecondIndexSwapping,
                                                                                         listOfNeighbour[0]) # Odległości pomiędzy sąsiadami indeksów, których zamieniliśmy
        return (newRoute-previousRoute)
    firstIdx = firstIndexSwapping
    secIdx = SecondIndexSwapping
    if SecondIndexSwapping < firstIndexSwapping:
        firstIdx = SecondIndexSwapping
        secIdx = firstIndexSwapping # Zamiana kolejności indeksów aby spełnić warunek funkcji calculateRouteChange...
    distanceChange = calculate_route_change_for_neighbour(distanceMatrix, cityOrder, firstIdx, secIdx)
    return (distanceChange) # Zwracamy różnicę dystansu

def calculate_route_change_for_neighbour(distanceMatrix, cityOrder, i, j): # Funkcja licząca odległości sąsiadów, i < j!
    lenght = len(cityOrder)
    if i == 0 and j == lenght-1: # sprawdzamy czy nie zamieniamy -1 indeksu z 0
        length_before = (distanceMatrix[cityOrder[0], cityOrder[1]] +
                         distanceMatrix[cityOrder[-1], cityOrder[-2]])
        length_after = (distanceMatrix[cityOrder[0], cityOrder[-2]] +
                        distanceMatrix[cityOrder[-1], cityOrder[1]])
        return length_after - length_before


    # Długość przed zmianą
    length_before = (distanceMatrix[cityOrder[i-1], cityOrder[i]] + # sprawdzamy odległość (i) oraz jego lewego sąsiada wraz z odległością j z jego prawym sąsiadem
                     distanceMatrix[cityOrder[i+1], cityOrder[(i+2) % lenght]])

    # Długość po zmianie
    length_after = (distanceMatrix[cityOrder[i-1], cityOrder[i+1]] + # sprawdzamy odległość j z lewym sąsiadem i wraz z odległością itego miasta z prawym sąsiadem jtego miasta
                    distanceMatrix[cityOrder[i], cityOrder[(i+2) % lenght]])

    # Zwraca różnicę długości
    return length_after - length_before

def check_if_we_get_better_route_for_insertion(distanceMatrix: np.array,cityOrder: np.array,indexOfCityInCityOrder,indexToInsert): # Funkcja licząca różnicę odległości, gdybyśmy dokonali swappingu. Brak założeń z indeksami
    lenOfCityOrder = len(cityOrder)
    if if_swap_neighbour(lenOfCityOrder-1,indexOfCityInCityOrder,indexToInsert) == False: # Jeśli zamieniane miasta nie są sąsiadami, to:
        if indexOfCityInCityOrder < indexToInsert:
            lengthBefore = (distanceMatrix[cityOrder[indexOfCityInCityOrder], cityOrder[indexOfCityInCityOrder-1]]
                            + distanceMatrix[cityOrder[indexOfCityInCityOrder], cityOrder[(indexOfCityInCityOrder+1) % lenOfCityOrder]]
                            + distanceMatrix[cityOrder[indexToInsert], cityOrder[(indexToInsert+1) % lenOfCityOrder]])
            lengthAfter = (distanceMatrix[cityOrder[indexOfCityInCityOrder], cityOrder[indexToInsert]] +
                           distanceMatrix[cityOrder[indexOfCityInCityOrder], cityOrder[(indexToInsert+1) % lenOfCityOrder]] +
                           distanceMatrix[cityOrder[indexOfCityInCityOrder+1], cityOrder[(indexOfCityInCityOrder-1)]])
            return lengthAfter - lengthBefore # Zwracamy różnicę odległości
        lengthBefore = (distanceMatrix[cityOrder[indexOfCityInCityOrder], cityOrder[indexOfCityInCityOrder-1]] +
                        distanceMatrix[cityOrder[indexOfCityInCityOrder], cityOrder[(indexOfCityInCityOrder+1) % lenOfCityOrder]] +
                        distanceMatrix[cityOrder[indexToInsert], cityOrder[indexToInsert-1]])
        lengthAfter = (distanceMatrix[cityOrder[indexOfCityInCityOrder], cityOrder[indexToInsert]] +
                       distanceMatrix[cityOrder[indexOfCityInCityOrder], cityOrder[indexToInsert-1]] +
                       distanceMatrix[cityOrder[(indexOfCityInCityOrder+1) % lenOfCityOrder], cityOrder[(indexOfCityInCityOrder-1)]])
        return lengthAfter - lengthBefore # Zwracamy różnicę odległości
    elif abs(indexOfCityInCityOrder-indexToInsert) == 1: # Jeśli zamieniane miasta są sąsiadami, to zamieniami kolejności indeksów, aby spełnić wymagania funkcji calculate_route_change_for_neighbour i ją wykonujemy
        firstIdx = indexOfCityInCityOrder
        secIdx = indexToInsert
        if indexToInsert < indexOfCityInCityOrder:
            firstIdx = indexToInsert
            secIdx = indexOfCityInCityOrder
        return calculate_route_change_for_neighbour(distanceMatrix, cityOrder, firstIdx, secIdx) # Zwracamy różnicę odległości
    return 0

# FUNKCJA DO SPRAWDZANIA RÓŻNICY ODLEGŁOŚCI DLA REVERSE (funkcja zwraca różnicę pomiędzy nową trasą a starą) WARUNEK KONIECZNY: firstIdX > secondIdx
def check_if_we_get_better_route_for_reverse(distanceMatrix: np.array, cityOrder: np.array, firstIdx, secondIdx):
    lenOfCityOrder = len(cityOrder) # Liczba miast w trasie

    if firstIdx == 0 and secondIdx == lenOfCityOrder - 1: # Warunek sprawdzający czy zamieniamy pierwsze z ostatnim miastem, jeśli tak to jest to ta sama odległość
        return 0
    if firstIdx - secondIdx == -1: # Warunek sprawdzający, czy wylosowano sąsiadów, jeśli tak, to zwraca nam wynik funkcji calculate_route_change_for_neighbour
        return calculate_route_change_for_neighbour(distanceMatrix, cityOrder, firstIdx, secondIdx)
    lengthBefore = (distanceMatrix[cityOrder[firstIdx], cityOrder[firstIdx - 1]] + # Jest to suma odległości pomiędzy itym miastem i sąsiadem po lewej oraz j miastem i sąsiadem po jego prawej
                    distanceMatrix[cityOrder[secondIdx], cityOrder[(secondIdx + 1) % lenOfCityOrder]])
    lengthAfter = (distanceMatrix[cityOrder[secondIdx], cityOrder[firstIdx - 1]] + # Jest to suma odległości pomiędzy jtym miastem i lewym sąsiadem itego miasta oraz itym miastem i prawym sąsiadem jtego miasta
                   distanceMatrix[cityOrder[firstIdx], cityOrder[(secondIdx + 1) % lenOfCityOrder]]) # Działamy tak, gdyż tylko te odległości się zmieniają przy odwracaniu.
    return lengthAfter - lengthBefore

# FUNKCJA DO ODWRACANIA
def reverse_method(order, firstIdx,secondIdx):
    reverseOrder = order.copy()
    reverseOrder[firstIdx:secondIdx + 1] = reverseOrder[firstIdx:secondIdx + 1][::-1]
    return reverseOrder

# FUNKCJA DO INSERCJI
def insert_method(order, index, insertion_index):
    element = order[index]
    new_order = np.delete(order, index)
    new_order = np.insert(new_order, insertion_index, element)
    return new_order


if __name__ == "__main__":
    readData = pd.read_csv("Miasta29.csv", sep=";", decimal=',')
    matrix = readData.iloc[:, 1:].astype(float).to_numpy()
    start_time = time.time()

    best_distance, best_route, best_global = run_simulated_annealing_multiple_times(
        matrix, num_runs=4, initial_temperature=10000, cooling_rate=0.003,
        num_iterations=100, acc_value=130000, min_temp=0.11, temp_red='fast', method="reverse",
        filename=get_result_file_path(f"Wyzarzanie_records_127.txt"))

    ### TEMP_RED = slow jest dla wolnej redukcji temperatury, fast dla szybkiej, wg wykładu dla wolnej redukcji temperatury liczba iteracji jest równa 1.
    ### METHOD : REVERSE, SWAP, ISERTION

    end_time = time.time()

    print("Najlepsza trasa na koncu:", best_route)
    print("Najlepsza odległość na koncu:", best_distance)
    print("Najlepsza globalna trasa:", best_global[1])
    print("Najlepsza globalna ogleglosc:", best_global[0])
    print("Czas wykonania:", end_time - start_time, "sekundy")
