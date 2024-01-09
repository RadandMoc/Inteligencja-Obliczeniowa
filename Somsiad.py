import numpy as np
import pandas as pd


def nearest_neighbor_tsp(distance_matrix, start_index):
    n = distance_matrix.shape[0]  # Liczba miast
    visited = [start_index]  # Lista odwiedzonych miast, zaczynamy od miasta startowego
    total_distance = 0  # Suma odległości

    # Konwersja macierzy odległości na typ zmiennoprzecinkowy
    distance_matrix = distance_matrix.astype(float)

    while len(visited) < n:
        # Utworzenie listy wszystkich indeksów
        all_indices = set(range(n))

        # Usunięcie indeksów już odwiedzonych
        remaining_indices = list(all_indices - set(visited))

        # Znalezienie indeksu najbliższego miasta
        next_index = np.argmin(distance_matrix[visited[-1], remaining_indices])
        next_city = remaining_indices[next_index]

        # Aktualizacja całkowitej odległości
        total_distance += distance_matrix[visited[-1], next_city]

        # Dodanie miasta do listy odwiedzonych
        visited.append(next_city)
    total_distance += distance_matrix[visited[-1], visited[0]]
    visited.append(start_index)

    return total_distance, visited


def FindMin(distance_matrix):
    listOfResult = []
    for i in range(distance_matrix.shape[0]):
        listOfResult.append(nearest_neighbor_tsp(distance_matrix, i))
    # return listOfResult
    return list(filter(lambda x: x[0] == minValueInList(listOfResult), listOfResult))

def FindMinForPlots(distance_matrix):
    listOfResult = []
    for i in range(distance_matrix.shape[0]):
        listOfResult.append(nearest_neighbor_tsp(distance_matrix, i)[0])
    return listOfResult
    #return list(filter(lambda x: x[0] == minValueInList(listOfResult), listOfResult))

def minValueInList(listOfResult):
    return min(listOfResult, key=lambda x: x[0])[0]


# distance_matrix = np.random.randint(1, 100, size=(100, 100))  # Losowe odległości między miastami
# np.fill_diagonal(distance_matrix, 0)
# readData=pd.read_csv("Dane_TSP_127.csv",sep=";")
readData = pd.read_csv("Dane_TSP_48.csv", sep=";", decimal=",")

distance_matrix = readData.iloc[:, 1:].astype(float).to_numpy()

print(FindMin(distance_matrix))