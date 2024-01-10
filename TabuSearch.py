import sys

import numpy as np
import pandas as pd
import random
import copy
import time
from tqdm import tqdm
from enum import Enum

#create enum Method with "reverse" and "swap" values
class Method(Enum):
    Swap = 1
    Insertion = 2
    Reverse = 3

def get_difference(meth, new_r, current_r, distance_m, current_dist, idx1, idx2):
    if (meth == Method.Swap):
        neighbours = get_cities(new_r, idx1, idx2)  # Biorę sąsiadów indeksów
        new_distance = (current_dist +
                        check_if_we_get_better_route_swapping(distance_m, current_r, neighbours, idx1, idx2))  # Liczę dystans gdybym zamienił miasta
        new_route = swap_cities(new_r, idx1, idx2)  # Nowa trasa po swappingu
        return new_distance, new_route
    elif (meth == "insertion"):
        new_distance = (current_dist +
                        check_if_we_get_better_route_insertion(distance_m, current_r, idx1,
                                                            idx2))  # Liczę dystans gdybym zmienił kolejność miast
        new_route = insert_method(new_r, idx1, idx2)  # Nowa trasa po insercji
        return new_distance, new_route
    else:
        if (idx1 > idx2):
            idx1, idx2 = idx2, idx1  # Zamieniam indeksy, aby spełnić warunek funkcji checkIfGetBetter...
        new_distance = (current_dist +
                        check_if_we_get_better_route_reverse(distance_m, current_r, idx1, idx2))
        new_route = reverseMethod(new_r, idx1, idx2)
        return new_distance, new_route

def get_cities(cityOrder: np.array, firstIdx: int, secondIdx: int): # Funkcja, której zadaniem jest zebranie sąsiadów dla dwóch indeksów
    def city(index):
        return cityOrder[index % len(cityOrder)]

    neighbourOfFirstIndex = np.array([city(firstIdx - 1), city(
        firstIdx + 1)])  # Zbieram sąsiadów dla pierwszego miasta w funkcji city jest modulo by uniknac błedu wyjscia indeksu poza zakres dla firstidx=len(cityOrder)-1
    neighbourOfSecondIndex = np.array([city(secondIdx - 1), city(secondIdx + 1)])
    return neighbourOfFirstIndex, neighbourOfSecondIndex # Zwracamy dwóch sąsiadów dla dwóch indeksów (nr miast)
def calculate_route_distance(route, cities_df):
    sum = 0
    for i in range(-1,len(route)-1):
        sum = sum + cities_df[route[i],route[i+1]]
    return sum
def swap_cities(cityOrder,firstIdx,Secondidx):
    firstCity = cityOrder[firstIdx]
    cityOrder[firstIdx] = cityOrder[Secondidx]
    cityOrder[Secondidx] = firstCity
    return cityOrder

def check_if_we_get_better_route_swapping(distanceMatrix: np.array, cityOrder: np.array, listOfNeighbour, firstIndexSwapping,
                            SecondIndexSwapping): # Funkcja sprawdzająca różnice odległości gdyby dokonał się swapping
    if ifSwapNeighbour(len(cityOrder) - 1, firstIndexSwapping, SecondIndexSwapping) == False: # Sprawdzamy czy zamieniamy sąsiadów
        previousRoute = checkRouteWithNeighbour(distanceMatrix, cityOrder, firstIndexSwapping,
                                                listOfNeighbour[0]) + checkRouteWithNeighbour(distanceMatrix, cityOrder,
                                                                                              SecondIndexSwapping,
                                                                                              listOfNeighbour[1]) # Odległości pomiędzy sąsiadami indeksów, których nie zamieniamy jeszcze
        newRoute = checkRouteWithNeighbour(distanceMatrix, cityOrder, firstIndexSwapping,
                                           listOfNeighbour[1]) + checkRouteWithNeighbour(distanceMatrix, cityOrder,
                                                                                         SecondIndexSwapping,
                                                                                         listOfNeighbour[0]) # Odległości pomiędzy sąsiadami indeksów, których zamieniliśmy
        return (newRoute-previousRoute)
    firstIdx = firstIndexSwapping
    secIdx = SecondIndexSwapping
    if SecondIndexSwapping < firstIndexSwapping:
        firstIdx = SecondIndexSwapping
        secIdx = firstIndexSwapping # Zamiana kolejności indeksów aby spełnić warunek funkcji calculateRouteChange...
    distanceChange = calculateRouteChangeForNeighbour(distanceMatrix, cityOrder, firstIdx, secIdx)
    return (distanceChange) # Zwracamy różnicę dystansu

def check_if_we_get_better_route_insertion(distanceMatrix: np.array,cityOrder: np.array,indexOfCityInCityOrder,indexToInsert): # Funkcja licząca różnicę odległości, gdybyśmy dokonali swappingu. Brak założeń z indeksami
    lenOfCityOrder = len(cityOrder)
    if ifSwapNeighbour(lenOfCityOrder-1,indexOfCityInCityOrder,indexToInsert) == False: # Jeśli zamieniane miasta nie są sąsiadami, to:
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
    elif abs(indexOfCityInCityOrder-indexToInsert) == 1: # Jeśli zamieniane miasta są sąsiadami, to zamieniami kolejności indeksów, aby spełnić wymagania funkcji calculateRouteChangeForNeighbour i ją wykonujemy
        firstIdx = indexOfCityInCityOrder
        secIdx = indexToInsert
        if indexToInsert < indexOfCityInCityOrder:
            firstIdx = indexToInsert
            secIdx = indexOfCityInCityOrder
        return calculateRouteChangeForNeighbour(distanceMatrix, cityOrder, firstIdx, secIdx) # Zwracamy różnicę odległości
    return 0

def insert_method(order, index, insertion_index):
    element = order[index]
    new_order = np.delete(order, index)
    new_order = np.insert(new_order, insertion_index, element)
    return new_order

def check_if_we_get_better_route_reverse(distanceMatrix: np.array, cityOrder: np.array, firstIdx, secondIdx):
    lenOfCityOrder = len(cityOrder) # Liczba miast w trasie

    if firstIdx == 0 and secondIdx == lenOfCityOrder - 1: # Warunek sprawdzający czy zamieniamy pierwsze z ostatnim miastem, jeśli tak to jest to ta sama odległość
        return 0
    if firstIdx - secondIdx == -1: # Warunek sprawdzający, czy wylosowano sąsiadów, jeśli tak, to zwraca nam wynik funkcji calculateRouteChangeForNeighbour
        return calculateRouteChangeForNeighbour(distanceMatrix, cityOrder, firstIdx, secondIdx)
    lengthBefore = (distanceMatrix[cityOrder[firstIdx], cityOrder[firstIdx - 1]] + # Jest to suma odległości pomiędzy itym miastem i sąsiadem po lewej oraz j miastem i sąsiadem po jego prawej
                    distanceMatrix[cityOrder[secondIdx], cityOrder[(secondIdx + 1) % lenOfCityOrder]])
    lengthAfter = (distanceMatrix[cityOrder[secondIdx], cityOrder[firstIdx - 1]] + # Jest to suma odległości pomiędzy jtym miastem i lewym sąsiadem itego miasta oraz itym miastem i prawym sąsiadem jtego miasta
                   distanceMatrix[cityOrder[firstIdx], cityOrder[(secondIdx + 1) % lenOfCityOrder]]) # Działamy tak, gdyż tylko te odległości się zmieniają przy odwracaniu.
    return lengthAfter - lengthBefore

# FUNKCJA DO ODWRACANIA
def reverseMethod(order, firstIdx,secondIdx):
    reverseOrder = order.copy()
    reverseOrder[firstIdx:secondIdx + 1] = reverseOrder[firstIdx:secondIdx + 1][::-1]
    return reverseOrder


def ifSwapNeighbour(lenght, firstIndexSwapping, SecondIndexSwapping): # Sprawdzamy czy indeksy są sąsiadami
    return abs(SecondIndexSwapping - firstIndexSwapping) <= 1 or (
                firstIndexSwapping == 0 and SecondIndexSwapping == lenght) or (
                SecondIndexSwapping == 0 and firstIndexSwapping == lenght)

def checkRouteWithNeighbour(distanceMatrix, cityOrder, index, neighbourCities): # Liczy nam trasę indeksu z dwoma sąsiadami
    return distanceMatrix[cityOrder[index], neighbourCities[0]] + distanceMatrix[cityOrder[index], neighbourCities[1]]

def calculateRouteChangeForNeighbour(distanceMatrix, cityOrder, i, j): # Funkcja licząca odległości sąsiadów, i < j!
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

# Funkcja obliczająca odległość euklidesową między dwoma miastami
def calculate_euclidean_distance(city1, city2, cities_df):
    dist = cities_df[city1, city2]
    return dist

# Generowanie sąsiedztwa poprzez zamianę dwóch miast w trasie
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

# Wybieranie najlepszego sąsiedztwa spośród sąsiadów, które nie są na liście tabu
def calculate_distance(neighborhood_moves, current_route, current_distance, cities_df, method):
    # distance = 0
    route = neighborhood_moves[0]
    idx1 = neighborhood_moves[1][0]
    idx2 = neighborhood_moves[1][1]
    city1 = route[idx1]
    city2 = route[idx2]
    new_route = np.copy(current_route)
    new_distance, new_route = get_difference(method, new_route, current_route, cities_df, current_distance, idx1, idx2)
    return new_distance, new_route, idx1, idx2


# Aktualizacja listy tabu
def update_tabu_list(tabu_list, new_solution, tabu_tenure):
    tabu_list.append(new_solution)
    if len(tabu_list) > tabu_tenure:
        tabu_list.pop(0)

def tabu_search(csv_file, output_file, iterations, critic_counter, method, tabu_tenure=7):
    start_time = time.time()  # Początek pomiaru czasu
    read_data = pd.read_csv(csv_file, sep=";", decimal=',')
    cities_df = read_data.iloc[:, 1:].astype(float).to_numpy()

    num_cities = len(cities_df)
    current_solution = list(range(0, num_cities))
    random.shuffle(current_solution)
    overall_best_solution = current_solution
    overall_best_distance = calculate_route_distance(overall_best_solution, cities_df)

    tabu_list = []

    for n in tqdm(range(iterations)):
        neighbor_current_distance = sys.maxsize
        best_distance_in_tabu = sys.maxsize
        best_solution_in_tabu = []
        best_solution = current_solution
        best_distance = calculate_route_distance(current_solution, cities_df)
        neighborhood_moves = generate_neighborhood(best_solution)
        best_move = []
        best_neighbor = None
        for neighbor in neighborhood_moves:
            new_distance, new_route, idx1, idx2 = calculate_distance(neighbor, best_solution, best_distance, cities_df, method)
            if (idx1 > idx2):
                idx1, idx2 = idx2, idx1
            neighbor_move = [idx1, idx2]

            if (neighbor_move in tabu_list):
                if(new_distance < best_distance_in_tabu):
                    best_distance_in_tabu = new_distance
                    best_solution_in_tabu = new_route
                continue
            if (new_distance < neighbor_current_distance):  # Sprawdzam czy dystans jest mniejszy
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
                #print(f"First condition: {best_distance}\nList: {tabu_list}\nMove: {best_move}")
            elif best_distance_in_tabu < overall_best_distance: #and random.random() < 0.2:
                best_solution = best_solution_in_tabu
                best_distance = best_distance_in_tabu
                update_tabu_list(tabu_list, [], tabu_tenure) #pusta tablica po to, żeby odjęło 1 od kolejki tabu
                #print(f"Second condition: {best_distance}\nList: {tabu_list}\nMove: {best_move}")
            else:
                best_solution = current_solution
                best_distance = current_distance
                update_tabu_list(tabu_list, best_move, tabu_tenure)
                #print(f"Third condition: {best_distance}\nList: {tabu_list}\nMove: {best_move}")

        if(best_distance < overall_best_distance):
            overall_best_distance = best_distance
            overall_best_solution = best_solution
    # Zapisz wynik do pliku tx
    #end timer
    end_time = time.time()  # Koniec pomiaru czasu
    elapsed_time = end_time - start_time
    with open(output_file, "a") as file:
        file.write(f"----------------\nFile: {csv_file}\nIterations: {iterations}\nCritic counter: {critic_counter}\nMethod: {method}\nTabu tenure: {tabu_tenure}\nBest solution: {overall_best_solution}\nDistance: {calculate_route_distance(overall_best_solution, cities_df)}\nTime: {elapsed_time}\n")
        print(f"----------------\nFile: {csv_file}\nIterations: {iterations}\nCritic counter: {critic_counter}\nMethod: {method}\nTabu tenure: {tabu_tenure}\nBest solution: {overall_best_solution}\nDistance: {calculate_route_distance(overall_best_solution, cities_df)}\nTime: {elapsed_time}\n")
    return overall_best_solution

# Przykładowe użycie
# best_solution = tabu_search('Dane_TSP_127.csv', "ResultsTabuSearch.txt",iterations=400, critic_counter =500, method= Method.Insertion, tabu_tenure=100)
best_solution = tabu_search('Dane_TSP_48.csv', "ResultsTabuSearch.txt",iterations=10000, critic_counter =500, method= Method.Reverse, tabu_tenure=200)