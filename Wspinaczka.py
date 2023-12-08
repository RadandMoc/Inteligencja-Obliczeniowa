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
    
    
def swapCities(cityOrder):
    indexOfTable = random.sample(range(0, len(cityOrder)), 2)
    newCityOrder = np.copy(cityOrder) 
    firstCity = cityOrder[indexOfTable[0]]
    newCityOrder[indexOfTable[0]] = cityOrder[indexOfTable[1]]
    newCityOrder[indexOfTable[1]] = firstCity
    return (newCityOrder,indexOfTable[0],indexOfTable[1])


def checkIfWeGetBetterRoute(distanceMatrix,cityOrder,newcityOrder
    ,firstIndexSwapping,SecondIndexSwapping):
    previousRoute = checkRouteWithNeighbour(distanceMatrix,cityOrder,firstIndexSwapping)+checkRouteWithNeighbour(distanceMatrix,cityOrder,SecondIndexSwapping)
    newRoute = checkRouteWithNeighbour(distanceMatrix,newcityOrder,firstIndexSwapping)+checkRouteWithNeighbour(distanceMatrix,newcityOrder,SecondIndexSwapping)
    if newRoute < previousRoute:
        return newcityOrder
    return cityOrder
    
        
def ifSwapNeighbour(firstIndexSwapping,SecondIndexSwapping):
    return abs(SecondIndexSwapping-firstIndexSwapping) <= 1
    

def checkRouteWithNeighbour(distanceMatrix,cityOrder,index):
    lenght = len(cityOrder)
    return distanceMatrix[cityOrder[index],cityOrder[(index-1+lenght)%lenght]]+distanceMatrix[cityOrder[index],cityOrder[(index+1+lenght)%lenght]]
 
"""
def checkRouteWithNeighbour(distanceMatrix,cityOrder,index):
    if index > 0 and index < len(cityOrder)-1:
        return distanceMatrix[cityOrder[index],cityOrder[index+1]]+distanceMatrix[cityOrder[index],cityOrder[index-1]]
    elif index == 0:
        return distanceMatrix[cityOrder[index],cityOrder[-1]]+distanceMatrix[cityOrder[index],cityOrder[index+1]]
    return distanceMatrix[cityOrder[index],cityOrder[index-1]]+distanceMatrix[cityOrder[index],cityOrder[0]]
"""

def ClimbingAlghoritmBySwapping(distanceMatrix,howManyIteration):
    cityOrder = getRandomRouteCities(distance_matrix.shape[0]) #ile wierszy
    
    for i in range(howManyIteration):
        newCityOrder = swapCities(cityOrder)
        cityOrder = checkIfWeGetBetterRoute(distanceMatrix,cityOrder,newCityOrder[0],newCityOrder[1],newCityOrder[2])
    return cityOrder    
    
    
def getSumOfCities(distanceMatrix,cityOrders):
    sum = 0
    for i in range(-1,len(cityOrders)-1):
        sum = sum + distanceMatrix[cityOrders[i],cityOrders[i+1]]
    return sum
        
    
"""
Not optimal function to show difference between two alghoritm
"""
def ClimbingAlghoritmBySwappingNotOptimal(distanceMatrix,howManyIteration):
    cityOrder = getRandomRouteCities(distance_matrix.shape[0]) #ile wierszy
    for i in range(howManyIteration):
        newcityOrder = swapCities(cityOrder)
        cityOrder = checkIfWeGetBetterRouteNotOptimal(distanceMatrix,cityOrder,newcityOrder[0],newcityOrder[1],newcityOrder[2])
    return cityOrder  
    
    
def checkIfWeGetBetterRouteNotOptimal(distanceMatrix,cityOrder,newcityOrder
    ,firstIndexSwapping,SecondIndexSwapping):
    previousRoute = getSumOfCities(distanceMatrix,cityOrder)
    newRoute = getSumOfCities(distanceMatrix,newcityOrder)
    if newRoute < previousRoute:
        return newcityOrder
    return cityOrder





readData=pd.read_csv("Dane_TSP_127.csv",sep=";")
readData = ChangeCommaToPoint(readData)
distance_matrix = readData.iloc[:,1:].astype(float).to_numpy()
start_time = time.time()

dobry = ClimbingAlghoritmBySwapping(distance_matrix,1000000)
print(dobry)
print(getSumOfCities(distance_matrix,dobry))
# The block of code to time
# [Your code here]
#[20, 4, 8, 11, 5, 27, 7, 26, 23, 12, 0, 25, 28, 2, 1, 9, 3, 14, 18, 15, 22, 6, 24, 10, 21, 13, 16, 17, 19]
#2387
# Current time after the block of code
end_time = time.time()
print(end_time-start_time)


start_time = time.time()

"""dobry = ClimbingAlghoritmBySwappingNotOptimal(distance_matrix,100000)
# The block of code to time
# [Your code here]

# Current time after the block of code
end_time = time.time()
print(end_time-start_time)
"""

