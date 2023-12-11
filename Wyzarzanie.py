import numpy as np
import pandas as pd
import random
import math
import copy
import time

def ChangeCommaToPoint(text):
    df_skopiowany = text.copy()  # Tworzymy kopię dataframe, aby nie zmieniać oryginalnego obiektu

    # Iterujemy po każdej komórce DataFrame i zamieniamy przecinki na kropki
    for kolumna in df_skopiowany.columns:
        if df_skopiowany[kolumna].dtype == 'object':  # Sprawdzamy tylko kolumny zawierające tekst
            df_skopiowany[kolumna] = df_skopiowany[kolumna].astype(str).str.replace(',', '.')

    return df_skopiowany

def getRandomRouteCities(numberOfCities):
    return random.sample(range(0, numberOfCities), numberOfCities)

def calculateTotalDistance(route, distance_matrix):
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += distance_matrix[route[i], route[i + 1]]
    total_distance += distance_matrix[route[-1], route[0]]  # Powrót do pierwszego miasta
    return total_distance

def acceptanceProbability(old_distance, new_distance, temperature):
    if new_distance < old_distance:
        return 1.0
    return math.exp((old_distance - new_distance) / temperature)

def simulatedAnnealing(distance_matrix, initial_temperature=1000, cooling_rate=0.003, num_iterations=100000):
    current_route = getRandomRouteCities(len(distance_matrix))
    current_distance = calculateTotalDistance(current_route, distance_matrix)
    best_route = copy.deepcopy(current_route)
    best_distance = current_distance

    temperature = initial_temperature

    for iteration in range(num_iterations):
        new_route = copy.deepcopy(current_route)
        # Przestawienie dwóch losowych miast
        idx1, idx2 = random.sample(range(len(new_route)), 2)
        new_route[idx1], new_route[idx2] = new_route[idx2], new_route[idx1]

        new_distance = calculateTotalDistance(new_route, distance_matrix)

        # Akceptacja nowej trasy z pewnym prawdopodobieństwem
        if acceptanceProbability(current_distance, new_distance, temperature) > random.random():
            current_route = copy.deepcopy(new_route)
            current_distance = new_distance

        # Aktualizacja najlepszej trasy, jeśli znaleziono krótszą trasę
        if current_distance < best_distance:
            best_route = copy.deepcopy(current_route)
            best_distance = current_distance

        # Schładzanie temperatury
        temperature *= (1 - cooling_rate)

    return best_route, best_distance

# Wczytanie danych
readData = pd.read_csv("Dane_TSP_127.csv", sep=";")
readData = ChangeCommaToPoint(readData)
distance_matrix = readData.iloc[:, 1:].astype(float).to_numpy()

# Uruchomienie algorytmu symulowanego wyżarzania
start_time = time.time()
best_route, best_distance = simulatedAnnealing(distance_matrix)
end_time = time.time()

print("Najlepsza trasa:", best_route)
print("Najlepsza odległość:", best_distance)
print("Czas wykonania:", end_time - start_time, "sekundy")
