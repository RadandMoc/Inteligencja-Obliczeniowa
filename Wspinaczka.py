import numpy as np
import pandas as pd
import random
import math
import copy
import time
import datetime

def ChangeCommaToPoint(text):
    df_skopiowany = text.copy()  # Tworzymy kopię dataframe, aby nie zmieniać oryginalnego obiektu
    
    # Iterujemy po każdej komórce DataFrame i zamieniamy przecinki na kropki
    for kolumna in df_skopiowany.columns:
        if df_skopiowany[kolumna].dtype == 'object':  # Sprawdzamy tylko kolumny zawierające tekst
            df_skopiowany[kolumna] = df_skopiowany[kolumna].astype(str).str.replace(',', '.')
    
    return df_skopiowany


def getRandomRouteCities(numberOfCities):
    return random.sample(range(0, numberOfCities), numberOfCities)
   
    

def swapCities(cityOrder,firstIdx,Secondidx):
    firstCity = cityOrder[firstIdx]
    cityOrder[firstIdx] = cityOrder[Secondidx]
    cityOrder[Secondidx] = firstCity
    return cityOrder


def GetCities(cityOrder,firstIdx,secondIdx):
    def city(index):
        return cityOrder[index % len(cityOrder)]   
    neighbourOfFirstIndex = [city(firstIdx-1),city(firstIdx+1)]
    neighbourOfSecondIndex = [city(secondIdx-1),city(secondIdx+1)]
    return neighbourOfFirstIndex, neighbourOfSecondIndex
    
    
    
def checkIfWeGetBetterRoute(
    distanceMatrix,
    cityOrder,
    listOfNeighbour,
    firstIndexSwapping,
    SecondIndexSwapping
 
    ):
    previousRoute = checkRouteWithNeighbour(distanceMatrix,cityOrder,firstIndexSwapping,listOfNeighbour[0])+checkRouteWithNeighbour(distanceMatrix,cityOrder,SecondIndexSwapping,listOfNeighbour[1])
    newRoute = checkRouteWithNeighbour(distanceMatrix,cityOrder,firstIndexSwapping,listOfNeighbour[1])+checkRouteWithNeighbour(distanceMatrix,cityOrder,SecondIndexSwapping,listOfNeighbour[0])
    if newRoute < previousRoute:
        return swapCities(cityOrder,firstIndexSwapping,SecondIndexSwapping)
    return cityOrder
    
        
def ifSwapNeighbour(firstIndexSwapping,SecondIndexSwapping):
    return abs(SecondIndexSwapping-firstIndexSwapping) <= 1
    

def checkRouteWithNeighbour(distanceMatrix,cityOrder,index,neighbourCities):
    return distanceMatrix[cityOrder[index],neighbourCities[0]]+distanceMatrix[cityOrder[index],neighbourCities[1]]
 
"""
def checkRouteWithNeighbour(distanceMatrix,cityOrder,index):
    if index > 0 and index < len(cityOrder)-1:
        return distanceMatrix[cityOrder[index],cityOrder[index+1]]+distanceMatrix[cityOrder[index],cityOrder[index-1]]
    elif index == 0:
        return distanceMatrix[cityOrder[index],cityOrder[-1]]+distanceMatrix[cityOrder[index],cityOrder[index+1]]
    return distanceMatrix[cityOrder[index],cityOrder[index-1]]+distanceMatrix[cityOrder[index],cityOrder[0]]
"""

def ClimbingAlghoritmBySwapping(distanceMatrix,cityOrder,howManyIteration,cities):
    for i in range(howManyIteration):
        #newCityOrder = swapCities(cityOrder)
        indexOfTable = random.sample(cities, 2)
        getNeighbourCities = GetCities(cityOrder,indexOfTable[0],indexOfTable[1])
        cityOrder = checkIfWeGetBetterRoute(distanceMatrix,cityOrder,getNeighbourCities,indexOfTable[0],indexOfTable[1])
    return cityOrder    
    
    
def getSumOfCities(distanceMatrix,cityOrders):
    sum = 0
    for i in range(-1,len(cityOrders)-1):
        sum = sum + distanceMatrix[cityOrders[i],cityOrders[i+1]]
    return sum
        

def makeIteration(repetition,distanceMatrix,howManyIteration,acceptanceValue):
    for i in range (1):
        cityOrder = getRandomRouteCities(distance_matrix.shape[0]) #ile wierszy
     
        cities = range(len(cityOrder))
        dobry = ClimbingAlghoritmBySwapping(distance_matrix,cityOrder,howManyIteration,cities)
        wynik = getSumOfCities(distance_matrix,dobry)
        print(dobry)
        print(wynik)
        if wynik < acceptanceValue:
            print(dobry)
            print(wynik)
            actual_datetime = datetime.datetime.now()
            date_Format = "%Y-%m-%d-%H-%M-%S-%f"
            filename = f"plik_{actual_datetime.strftime(date_Format)}.txt"
            with open(filename, 'w') as resultFile:
                for element in dobry:
                    resultFile.write(str(element) + ' ')
                resultFile.write(str(wynik))

"""
Not optimal function to show difference between two alghoritm
"""
def ClimbingAlghoritmBySwappingNotOptimal(distanceMatrix,howManyIteration):
    cityOrder = getRandomRouteCities(distance_matrix.shape[0]) #ile wierszy
    for i in range(howManyIteration):
        #newcityOrder = swapCities(cityOrder)
        
        cityOrder = checkIfWeGetBetterRouteNotOptimal(distanceMatrix,cityOrder,newcityOrder[0],newcityOrder[1],newcityOrder[2])
    return cityOrder  
    
    
def checkIfWeGetBetterRouteNotOptimal(distanceMatrix,cityOrder,getNeighbourCities
    ,firstIndexSwapping,SecondIndexSwapping):
    previousRoute = getSumOfCities(distanceMatrix,cityOrder)
    newRoute = getSumOfCities(distanceMatrix,newcityOrder)
    if newRoute < previousRoute:
        return newcityOrder
    return cityOrder





#readData=pd.read_csv("Miasta29.csv",sep=";")
readData=pd.read_csv("Dane_TSP_127.csv",sep=";")
readData = ChangeCommaToPoint(readData)
distance_matrix = readData.iloc[:,1:].astype(float).to_numpy()
start_time = time.time()



makeIteration(1,distance_matrix,100000,160000)





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

