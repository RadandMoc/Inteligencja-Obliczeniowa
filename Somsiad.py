import numpy as np
import pandas as pd


def nearest_neighbor_tsp(distance_matrix, start_index):
    n = distance_matrix.shape[0] 
    visited = [start_index] 
    total_distance = 0 

    distance_matrix = distance_matrix.astype(float)
    all_indices = set(range(n))
    while len(visited) < n:
        remaining_indices = list(all_indices - set(visited))
        next_index = np.argmin(distance_matrix[visited[-1], remaining_indices])
        next_city = remaining_indices[next_index]
        total_distance += distance_matrix[visited[-1], next_city]
        visited.append(next_city)
    total_distance += distance_matrix[visited[-1], visited[0]]
    return total_distance, visited


def FindMin(distance_matrix):
    listOfResult = []
    for i in range(distance_matrix.shape[0]):
        listOfResult.append(nearest_neighbor_tsp(distance_matrix, i))
    return list(filter(lambda x: x[0] == minValueInList(listOfResult), listOfResult))

def minValueInList(listOfResult):
    return min(listOfResult, key=lambda x: x[0])[0]



#readData=pd.read_csv("Dane_TSP_127.csv",sep=";")
readData = pd.read_csv("Dane_TSP_48.csv", sep=";", decimal=",")
#readData = pd.read_csv("Dane_TSP_76.csv", sep=";", decimal=",")


distance_matrix = readData.iloc[:, 1:].astype(float).to_numpy()

print(FindMin(distance_matrix))