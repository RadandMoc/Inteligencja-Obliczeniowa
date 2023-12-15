import numpy as np
import pandas as pd
import random
import math
import copy
import time
import sys

def ChangeCommaToPoint(text):
    df_skopiowany = text.copy()
    for kolumna in df_skopiowany.columns:
        if df_skopiowany[kolumna].dtype == 'object':
            df_skopiowany[kolumna] = df_skopiowany[kolumna].astype(str).str.replace(',', '.')
    return df_skopiowany


def save_data(best_overall, acc_value,filename=f"Wyzarzanie_records.txt"):
        with open(filename, 'a') as resultFile:
            resultFile.write("\n" + "=" * 25 + "\n")  # Dodanie kreski oddzielającej dane
            for element in best_overall[1]:
                resultFile.write(str(element+1) + ' ')
            resultFile.write(str(best_overall[0]))

def getRandomRouteCities(numberOfCities):
    return random.sample(range(0, numberOfCities), numberOfCities)

def getSumOfCities(cityOrders, distanceMatrix):
    sum = 0
    for i in range(-1,len(cityOrders)-1):
        sum = sum + distanceMatrix[cityOrders[i],cityOrders[i+1]]
    return sum

def swapCities(cityOrder,firstIdx,Secondidx):
    firstCity = cityOrder[firstIdx]
    cityOrder[firstIdx] = cityOrder[Secondidx]
    cityOrder[Secondidx] = firstCity
    return cityOrder

def acceptanceProbability(old_distance, new_distance, temperature):
    if new_distance < old_distance:
        return 1.0
    return math.exp((old_distance - new_distance) / temperature)

def simulatedAnnealing(distance_matrix, initial_temperature, cooling_rate, num_iterations, min_temp, best_res, method="swap"):
    current_route = getRandomRouteCities(len(distance_matrix))
    current_distance = getSumOfCities(current_route, distance_matrix)
    best = best_res
    global_best_list = []
    new_distance = 0

    temperature = initial_temperature

    while(temperature>min_temp):
        for iteration in range(num_iterations):
            new_route = np.copy(current_route)

            if (method == "swap"):
                idx1, idx2 = random.sample(range(len(new_route)), 2)
                neighbours = GetCities(new_route, idx1, idx2)
                new_distance = current_distance + checkIfWeGetBetterRoute(distance_matrix, current_route, neighbours, idx1, idx2)
                new_route = swapCities(new_route, idx1, idx2)


            #new_distance = getSumOfCities(new_route, distance_matrix)

            if acceptanceProbability(current_distance, new_distance, temperature) > random.random():
                current_route = np.copy(new_route)
                current_distance = new_distance

            if current_distance < best:
                global_best_route = np.copy(current_route)
                global_best_list = [current_distance, global_best_route]
                best = current_distance

        temperature *= (1 - cooling_rate)

    return current_route, current_distance, global_best_list

def runSimulatedAnnealingMultipleTimes(distance_matrix, num_runs, initial_temperature, cooling_rate, num_iterations, min_temp, acc_value, best_result=sys.maxsize):
    best_finished = [sys.maxsize, 21212121] # Najlepsza odległość pod koniec
    best_overall = [sys.maxsize, 21212121] # Najlepsza odległość ogólnie (może być pod koniec, może być nie pod koniec)

    for i in range(num_runs):
        data_run = simulatedAnnealing(distance_matrix, initial_temperature, cooling_rate, num_iterations, min_temp, best_result)
        best_route, best_distance, glbestlist = data_run[0], data_run[1], data_run[2]
        print(best_distance)

        if best_distance < best_finished[0]:
            best_finished = [best_distance, best_route]
            best_result = best_distance

        if (len(glbestlist)>0 and best_overall[0] > glbestlist[0]):
            best_overall = glbestlist
            best_result = best_distance

        if (best_overall[0] < acc_value):
            save_data(best_overall, acc_value)

    return best_finished[0], best_finished[1], best_overall


def ifSwapNeighbour(lenght, firstIndexSwapping, SecondIndexSwapping):
    return abs(SecondIndexSwapping - firstIndexSwapping) <= 1 or (
                firstIndexSwapping == 0 and SecondIndexSwapping == lenght) or (
                SecondIndexSwapping == 0 and firstIndexSwapping == lenght)

def checkRouteWithNeighbour(distanceMatrix, cityOrder, index, neighbourCities):
    return distanceMatrix[cityOrder[index], neighbourCities[0]] + distanceMatrix[cityOrder[index], neighbourCities[1]]


def GetCities(cityOrder: np.array, firstIdx: int, secondIdx: int):
    def city(index):
        return cityOrder[index % len(cityOrder)]

    neighbourOfFirstIndex = np.array([city(firstIdx - 1), city(
        firstIdx + 1)])  # Zbieram sąsiadów dla pierwszego miasta w funkcji city jest modulo by uniknac błedu wyjscia indeksu poza zakres dla firstidx=len(cityOrder)-1
    neighbourOfSecondIndex = np.array([city(secondIdx - 1), city(secondIdx + 1)])
    return neighbourOfFirstIndex, neighbourOfSecondIndex


def checkIfWeGetBetterRoute(distanceMatrix: np.array, cityOrder: np.array, listOfNeighbour, firstIndexSwapping,
                            SecondIndexSwapping):
    if ifSwapNeighbour(len(cityOrder) - 1, firstIndexSwapping, SecondIndexSwapping) == False:
        previousRoute = checkRouteWithNeighbour(distanceMatrix, cityOrder, firstIndexSwapping,
                                                listOfNeighbour[0]) + checkRouteWithNeighbour(distanceMatrix, cityOrder,
                                                                                              SecondIndexSwapping,
                                                                                              listOfNeighbour[1])
        newRoute = checkRouteWithNeighbour(distanceMatrix, cityOrder, firstIndexSwapping,
                                           listOfNeighbour[1]) + checkRouteWithNeighbour(distanceMatrix, cityOrder,
                                                                                         SecondIndexSwapping,
                                                                                         listOfNeighbour[0])
        return (newRoute-previousRoute) # ujemne to git
    firstIdx = firstIndexSwapping
    secIdx = SecondIndexSwapping
    if SecondIndexSwapping < firstIndexSwapping:
        firstIdx = SecondIndexSwapping
        secIdx = firstIndexSwapping
    distanceChange = calculate_route_change_for_neighbour(distanceMatrix, cityOrder, firstIdx, secIdx)
    return (distanceChange)

def calculate_route_change_for_neighbour(distanceMatrix, cityOrder, i, j):
    lenght = len(cityOrder)
    if i == 0 and j == lenght-1:
        length_before = (distanceMatrix[cityOrder[0], cityOrder[1]] +
                        distanceMatrix[cityOrder[-1], cityOrder[-2]])
        length_after = (distanceMatrix[cityOrder[0], cityOrder[-2]]+
                        distanceMatrix[cityOrder[-1], cityOrder[1]])
        return length_after - length_before


    # Długość przed zmianą
    length_before = (distanceMatrix[cityOrder[i-1], cityOrder[i]] +
                     distanceMatrix[cityOrder[i+1], cityOrder[(i+2)%lenght]])

    # Długość po zmianie
    length_after = (distanceMatrix[cityOrder[i-1], cityOrder[i+1]] +
                    distanceMatrix[cityOrder[i], cityOrder[(i+2)%lenght]])

    # Zwraca różnicę długości
    return length_after - length_before







# Wczytanie danych
readData = pd.read_csv("Dane_TSP_127.csv", sep=";")
readData = ChangeCommaToPoint(readData)
distance_matrix = readData.iloc[:, 1:].astype(float).to_numpy()

# Uruchomienie algorytmu symulowanego wyżarzania 3000 razy
start_time = time.time()
best_distance, best_route, best_global = runSimulatedAnnealingMultipleTimes(distance_matrix, num_runs=4, initial_temperature=10000, cooling_rate=0.003, num_iterations=100, acc_value=2000,min_temp=0.11)
end_time = time.time()

print("Trasa na koncu:", best_route)
print("Odległość na koncu:", best_distance)

print("Najlepsza globalna trasa:", best_global[0])
print("Najlepsza globalna odległość:", best_global[1])
print("Czas wykonania:", end_time - start_time, "sekundy")
