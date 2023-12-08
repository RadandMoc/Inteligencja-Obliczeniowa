import numpy as np
import pandas as pd
import random
import math
import copy
import time


def getRandomRouteCities(numberOfCities):
    return random.sample(range(0, numberOfCities), numberOfCities)
    
    
def swapCities(cityOrder):
    indexOfTable = random.sample(range(0, len(cityOrder)), 2)
    newcityOrder = copy.deepcopy(cityOrder) 
    first = cityOrder[indexOfTable[0]]
    newcityOrder[indexOfTable[0]] = cityOrder[indexOfTable[1]]
    newcityOrder[indexOfTable[1]] = first
    return (newcityOrder,indexOfTable[0],indexOfTable[1])


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
	
def randomNumberAndDelete(array):
    print("kot")

def ClimbingAlghoritmBySwapping(distanceMatrix,howManyIteration):
    cityOrder = getRandomRouteCities(distance_matrix.shape[0]) #ile wierszy
    for i in range(howManyIteration):
        newcityOrder = swapCities(cityOrder)
        cityOrder = checkIfWeGetBetterRoute(distanceMatrix,cityOrder,newcityOrder[0],newcityOrder[1],newcityOrder[2])
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
distance_matrix = readData.iloc[:,1:].astype(float).to_numpy()
start_time = time.time()

dobry = ClimbingAlghoritmBySwapping(distance_matrix,100000)
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

dobry = ClimbingAlghoritmBySwappingNotOptimal(distance_matrix,100000)
# The block of code to time
# [Your code here]

# Current time after the block of code
end_time = time.time()
print(end_time-start_time)


