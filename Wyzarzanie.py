import numpy as np
import pandas as pd
import random
import math
import time
import sys

# FUNKCJA DO ZAPISU DO PLIKU
def saveData(best_overall, initial_temperature, cooling_rate, num_iterations, min_temp, temp_red, metoda, filename):
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
def getRandomRouteCities(numberOfCities):
    return random.sample(range(0, numberOfCities), numberOfCities)

# FUNKCJA LICZĄCA CAŁKOWITĄ SUMĘ TRAS, PRZYDATNA TYLKO RAZ, GDYŻ POTEM ODLEGŁOŚĆ LICZYMY NA PODSTAWIE RÓŻNIC
def getSumOfCities(cityOrders, distanceMatrix):
    sum = 0
    for i in range(-1,len(cityOrders)-1):
        sum = sum + distanceMatrix[cityOrders[i],cityOrders[i+1]]
    return sum

# FUNKCJA DO SWAPINGU
def swapCities(cityOrder,firstIdx,Secondidx):
    firstCity = cityOrder[firstIdx]
    cityOrder[firstIdx] = cityOrder[Secondidx]
    cityOrder[Secondidx] = firstCity
    return cityOrder

# FUNKCJA LICZĄCA PRAWDOPODOBIEŃSTWO ZAMIANY
def acceptanceProbability(old_distance, new_distance, temperature):
    if new_distance < old_distance:
        return 1.0
    return math.exp((old_distance - new_distance) / temperature)

# POJEDYNCZA ITERACJA SYMULOWANEGO WYŻARZANIA
def simulatedAnnealing(distance_matrix, temperature, cooling_rate, num_iterations, min_temp, temp_red, method="swap"):
    current_route = getRandomRouteCities(len(distance_matrix)) # Początkowa trasa - randomowa
    current_distance = getSumOfCities(current_route, distance_matrix) # Dystans początkowej trasy
    best_one_iteration = current_distance # Globalne najlepsze rozwiązanie w jednej iteracji
    best_one_route = np.copy(current_route)
    new_distance = 0

    if(temp_red == 'slow'):
        num_iterations = 1

    while(temperature>min_temp): # Pętla się wykonuje, aż temperatura nie będzie miała pożądanej temperatury
        for iteration in range(num_iterations): # Pętla wykonuje się określoną liczbę iteracji
            new_route = np.copy(current_route)
            idx1, idx2 = random.sample(range(len(new_route)), 2) # Losujemy 2 randomowe indexy od 0 do (długości trasy-1)

            if (method == "swap"):
                neighbours = getCities(new_route, idx1, idx2) # Biorę sąsiadów indeksów
                new_distance = (current_distance +
                                checkIfWeGetBetterRouteSwapping(distance_matrix, current_route, neighbours, idx1, idx2)) # Liczę dystans gdybym zamienił miasta
                new_route = swapCities(new_route, idx1, idx2) # Nowa trasa po swappingu
            elif (method == "insertion"):
                new_distance = (current_distance +
                                checkIfWeGetBetterRouteForInsertion(distance_matrix, current_route, idx1, idx2)) # Liczę dystans gdybym zmienił kolejność miast
                new_route = insertMethod(new_route, idx1, idx2) # Nowa trasa po insercji
            else:
                if (idx1 > idx2):
                    idx1, idx2 = idx2, idx1 # Zamieniam indeksy, aby spełnić warunek funkcji checkIfGetBetter...
                new_distance = (current_distance +
                                checkIfWeGetBetterRouteForReverse(distance_matrix, current_route, idx1, idx2))
                new_route = reverseMethod(new_route, idx1, idx2)


            if acceptanceProbability(current_distance, new_distance, temperature) > random.random(): # Jeśli prawdopodobieństwo wybrania trasy jest większe niż losowa liczba z zakresu 0 do 1, to:
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
def runSimulatedAnnealingMultipleTimes(distance_matrix, num_runs,
                                       initial_temperature, cooling_rate, num_iterations, min_temp, acc_value,
                                       filename, temp_red, method="swap"): # Funkcja wykonującaco symulowane wyżarzanie num_runs

    if(method != "swap" and method != "reverse" and method != "insertion"): # Sprawdzanie czy wprowadziliśmy istniejącą metodę zamiany
        print("Method doesn't exist.")
        return [0, 0, [0, 0]]

    best_finished = [sys.maxsize, []] # Najlepsza odległość pod koniec z num_runs razy wykonanego algorytmu wraz z trasą
    best_overall = [sys.maxsize, []] # Najlepsza odległość ogólnie z num_rns razy wykonanego algorytmu wraz z trasą (może być pod koniec, może być nie pod koniec)

    for i in range(num_runs):
        data_run = simulatedAnnealing(distance_matrix, initial_temperature, cooling_rate, num_iterations, min_temp, temp_red, method) # Wykonanie algorytmu symulowanego wyżarzania
        best_route, best_distance, glbestone = data_run[0], data_run[1], data_run[2] # Kolejno najlepsza odległość końcowa, najlepszy dystans końcowy , najlepsze rozwiązanie w całym okresie wykonywanego algorytmu, najlepsze rozwiązanie w okresie jednej iteracji
        print("Odległość na końcu:", best_distance) # Printuje nam końcowy dystans

        if best_distance < best_finished[0]: # Jeśli końcowy dystans jest lepszy od najlepszego końcowego dystansu, to:
            best_finished = [best_distance, best_route] # Przypisujemy do zmiennej best_finished dystans i trasę


        if (best_overall[0] > glbestone[0]): # Jeśli najlepsze rozwiązanie ogólne z iteracji algorytmu symulowanego wyżarzania jest najlepsze, (np. gdy rozwiązanie z 4 iteracji wykonania całego algorytmu jest lepsze niż rozwiązanie z 2 iteracji, które było do tego czasu najlepszym):
            best_overall = glbestone # Przypisuję do zmiennej best_overall listę posiadającą trasę wraz z odległością

        if (glbestone[0] < acc_value): # Jeśli najlepsza globalna odległość jest lepsza od tej, jaką ustalimy, to zapisujemy do pliku
            saveData(glbestone, initial_temperature=initial_temperature,
                     cooling_rate=cooling_rate, num_iterations=num_iterations,
                     min_temp=min_temp, temp_red=temp_red, metoda=method, filename=filename)

    return best_finished[0], best_finished[1], best_overall # Zwracamy najlepszą trasę końcową ze wszystkich wraz z trasą oraz zwracamy najlepsze globalne rozwiązanie


def ifSwapNeighbour(lenght, firstIndexSwapping, SecondIndexSwapping): # Sprawdzamy czy indeksy są sąsiadami
    return abs(SecondIndexSwapping - firstIndexSwapping) <= 1 or (
                firstIndexSwapping == 0 and SecondIndexSwapping == lenght) or (
                SecondIndexSwapping == 0 and firstIndexSwapping == lenght)

def checkRouteWithNeighbour(distanceMatrix, cityOrder, index, neighbourCities): # Liczy nam trasę indeksu z dwoma sąsiadami
    return distanceMatrix[cityOrder[index], neighbourCities[0]] + distanceMatrix[cityOrder[index], neighbourCities[1]]


def getCities(cityOrder: np.array, firstIdx: int, secondIdx: int): # Funkcja, której zadaniem jest zebranie sąsiadów dla dwóch indeksów
    def city(index):
        return cityOrder[index % len(cityOrder)]

    neighbourOfFirstIndex = np.array([city(firstIdx - 1), city(
        firstIdx + 1)])  # Zbieram sąsiadów dla pierwszego miasta w funkcji city jest modulo by uniknac błedu wyjscia indeksu poza zakres dla firstidx=len(cityOrder)-1
    neighbourOfSecondIndex = np.array([city(secondIdx - 1), city(secondIdx + 1)])
    return neighbourOfFirstIndex, neighbourOfSecondIndex # Zwracamy dwóch sąsiadów dla dwóch indeksów (nr miast)


def checkIfWeGetBetterRouteSwapping(distanceMatrix: np.array, cityOrder: np.array, listOfNeighbour, firstIndexSwapping,
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

def checkIfWeGetBetterRouteForInsertion(distanceMatrix: np.array,cityOrder: np.array,indexOfCityInCityOrder,indexToInsert): # Funkcja licząca różnicę odległości, gdybyśmy dokonali swappingu. Brak założeń z indeksami
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

# FUNKCJA DO SPRAWDZANIA RÓŻNICY ODLEGŁOŚCI DLA REVERSE (funkcja zwraca różnicę pomiędzy nową trasą a starą) WARUNEK KONIECZNY: firstIdX > secondIdx
def checkIfWeGetBetterRouteForReverse(distanceMatrix: np.array, cityOrder: np.array, firstIdx, secondIdx):
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

# FUNKCJA DO INSERCJI
def insertMethod(order, index, insertion_index):
    element = order[index]
    new_order = np.delete(order, index)
    new_order = np.insert(new_order, insertion_index, element)
    return new_order


readData = pd.read_csv("Dane_TSP_127.csv", sep=";", decimal=',')
matrix = readData.iloc[:, 1:].astype(float).to_numpy()

start_time = time.time()

best_distance, best_route, best_global = runSimulatedAnnealingMultipleTimes(
    matrix, num_runs=4, initial_temperature=10000, cooling_rate=0.003,
    num_iterations=1000, acc_value=130000, min_temp=0.11, temp_red='fast', method="reverse",
    filename=f"Wyzarzanie_records_127.txt")

### TEMP_RED = slow jest dla wolnej redukcji temperatury, fast dla szybkiej, wg wykładu dla wolnej redukcji temperatury liczba iteracji jest równa 1.
### METHOD : REVERSE, SWAP, ISERTION

end_time = time.time()

print("Najlepsza trasa na koncu:", best_route)
print("Najlepsza odległość na koncu:", best_distance)
print("Najlepsza globalna trasa:", best_global[1])
print("Najlepsza globalna ogleglosc:", best_global[0])
print("Czas wykonania:", end_time - start_time, "sekundy")


