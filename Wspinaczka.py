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


def GetCities(cityOrder :np.array,firstIdx :int,secondIdx :int):
    def city(index):
        return cityOrder[index % len(cityOrder)] 
    neighbourOfFirstIndex = np.array([city(firstIdx-1),city(firstIdx+1)]) #Zbieram sąsiadów dla pierwszego miasta w funkcji city jest modulo by uniknac błedu wyjscia indeksu poza zakres dla firstidx=len(cityOrder)-1
    neighbourOfSecondIndex = np.array([city(secondIdx-1),city(secondIdx+1)])
    return neighbourOfFirstIndex, neighbourOfSecondIndex


def checkIfWeGetBetterRoute(distanceMatrix: np.array,cityOrder: np.array,listOfNeighbour,firstIndexSwapping,SecondIndexSwapping):
    if ifSwapNeighbour(len(cityOrder)-1,firstIndexSwapping,SecondIndexSwapping) == False:
        previousRoute = checkRouteWithNeighbour(distanceMatrix,cityOrder,firstIndexSwapping,listOfNeighbour[0])+checkRouteWithNeighbour(distanceMatrix,cityOrder,SecondIndexSwapping,listOfNeighbour[1])
        newRoute = checkRouteWithNeighbour(distanceMatrix,cityOrder,firstIndexSwapping,listOfNeighbour[1])+checkRouteWithNeighbour(distanceMatrix,cityOrder,SecondIndexSwapping,listOfNeighbour[0])
    
        if newRoute < previousRoute:
            #print(listOfNeighbour)
            return (swapCities(cityOrder,firstIndexSwapping,SecondIndexSwapping),True)
        return (cityOrder,False)
    firstIdx = firstIndexSwapping
    secIdx = SecondIndexSwapping
    if SecondIndexSwapping < firstIndexSwapping:
        firstIdx = SecondIndexSwapping
        secIdx = firstIndexSwapping
    distanceChange = calculate_route_change_for_neighbour(distanceMatrix, cityOrder, firstIdx, secIdx)
    if distanceChange < 0:
        return (swapCities(cityOrder,firstIndexSwapping,SecondIndexSwapping),True)
    return (cityOrder,False)


        


def ifSwapNeighbour(lenght,firstIndexSwapping,SecondIndexSwapping):
    return abs(SecondIndexSwapping-firstIndexSwapping) <= 1 or (firstIndexSwapping==0 and SecondIndexSwapping==lenght) or (SecondIndexSwapping==0 and firstIndexSwapping==lenght)
    

def checkRouteWithNeighbour(distanceMatrix,cityOrder,index,neighbourCities):
    return distanceMatrix[cityOrder[index],neighbourCities[0]]+distanceMatrix[cityOrder[index],neighbourCities[1]]

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


 
 
"""
def checkRouteWithNeighbour(distanceMatrix,cityOrder,index):
    if index > 0 and index < len(cityOrder)-1:
        return distanceMatrix[cityOrder[index],cityOrder[index+1]]+distanceMatrix[cityOrder[index],cityOrder[index-1]]
    elif index == 0:
        return distanceMatrix[cityOrder[index],cityOrder[-1]]+distanceMatrix[cityOrder[index],cityOrder[index+1]]
    return distanceMatrix[cityOrder[index],cityOrder[index-1]]+distanceMatrix[cityOrder[index],cityOrder[0]]
"""

def ClimbingAlghoritmBySwapping(distanceMatrix,cityOrder,howManyIterationWithoutImprovement,cities):

    i = 0

    while i < howManyIterationWithoutImprovement:
        #newCityOrder = swapCities(cityOrder)
        indexOfTable = random.sample(cityOrder, 2)
        getNeighbourCities = GetCities(cityOrder,indexOfTable[0],indexOfTable[1])
        
        changeCityOrder = checkIfWeGetBetterRoute(distanceMatrix,cityOrder,getNeighbourCities,indexOfTable[0],indexOfTable[1])
        cityOrder = changeCityOrder[0]
        i = i+1
        if changeCityOrder[1] == True: #If change city order
            i = 0
    return cityOrder    
    
    
def getSumOfCities(distanceMatrix,cityOrders):
    sum = 0
    for i in range(-1,len(cityOrders)-1):
        sum = sum + distanceMatrix[cityOrders[i],cityOrders[i+1]]
    return sum
        

def makeIteration(repetition,distanceMatrix,howManyIterationWithoutImprovem,acceptanceValue):
    for i in range (repetition):
        cityOrder = getRandomRouteCities(distanceMatrix.shape[0]) #ile wierszy
        cities = range(len(cityOrder))
        finalCitiesOrder = ClimbingAlghoritmBySwapping(distanceMatrix,cityOrder,howManyIterationWithoutImprovem,cities)
        sumOfFinalResult = getSumOfCities(distanceMatrix,finalCitiesOrder)
      
        if sumOfFinalResult < acceptanceValue:
            
            actual_datetime = datetime.datetime.now()
            date_Format = "%Y-%m-%d-%H-%M-%S-%f"
            filename = f"plik_{actual_datetime.strftime(date_Format)}.txt"
            with open(filename, 'w') as resultFile:
                for element in finalCitiesOrder:
                    resultFile.write(str(element) + ' ')
                resultFile.write(str(sumOfFinalResult))


def ClimbingAlghoritmByInsertion(distanceMatrix,cityOrder,howManyIterationWithoutImprovement):
    i = 0
    while i < howManyIterationWithoutImprovement:
        indexOfCityAndIndexOfInsetrion = random.sample(cityOrder, 2)



def checkIfWeGetBetterRouteForInsertion(distanceMatrix: np.array,cityOrder: np.array,indexOfCityInCityOrder,indexToInsert):
    lenOfCityOrder = len(cityOrder)
    if ifSwapNeighbour(lenOfCityOrder-1,indexOfCityInCityOrder,indexToInsert) == False:
        print("kot")
        if indexOfCityInCityOrder < indexToInsert:
            lengthBefore = distanceMatrix[cityOrder[indexOfCityInCityOrder],cityOrder[indexOfCityInCityOrder-1]] + distanceMatrix[cityOrder[indexOfCityInCityOrder],cityOrder[(indexOfCityInCityOrder+1)%lenOfCityOrder]] + distanceMatrix[cityOrder[indexToInsert],cityOrder[(indexToInsert+1)%lenOfCityOrder]] 

            lengthAfter = distanceMatrix[cityOrder[indexOfCityInCityOrder],cityOrder[indexToInsert]] + distanceMatrix[cityOrder[indexOfCityInCityOrder],cityOrder[(indexToInsert+1)%lenOfCityOrder]] + distanceMatrix[cityOrder[indexOfCityInCityOrder+1],cityOrder[(indexOfCityInCityOrder-1)]] 
            return lengthAfter - lengthBefore
        lengthBefore = distanceMatrix[cityOrder[indexOfCityInCityOrder],cityOrder[indexOfCityInCityOrder-1]] + distanceMatrix[cityOrder[indexOfCityInCityOrder],cityOrder[(indexOfCityInCityOrder+1)%lenOfCityOrder]] + distanceMatrix[cityOrder[indexToInsert],cityOrder[indexToInsert-1]] 
        lengthAfter = distanceMatrix[cityOrder[indexOfCityInCityOrder],cityOrder[indexToInsert]] + distanceMatrix[cityOrder[indexOfCityInCityOrder],cityOrder[indexToInsert-1]] + distanceMatrix[cityOrder[indexOfCityInCityOrder+1],cityOrder[(indexOfCityInCityOrder-1)]] 
        return lengthAfter - lengthBefore
    elif abs(indexOfCityInCityOrder-indexToInsert) == 1:
        firstIdx = indexOfCityInCityOrder
        secIdx = indexToInsert
        if indexToInsert < indexOfCityInCityOrder:
            firstIdx = indexToInsert
            secIdx = indexOfCityInCityOrder
        return calculate_route_change_for_neighbour(distanceMatrix,cityOrder,firstIdx,secIdx)
    return 0
 



def checkIfWeGetBetterRouteForReverse(distanceMatrix: np.array,cityOrder: np.array,firstIdx,secondIdx):
    lenOfCityOrder = len(cityorder)
    if firstIdx==0 and secondIdx == lenOfCityOrder-1:
        return 0
    if firstIdx - secondIdx == -1:
        return calculate_route_change_for_neighbour(distanceMatrix,cityOrder,firstIdx,secondIdx)
    return distance_matrix[cityOrder[firstIdx],cityOrder[firstIdx-1]] + distance_matrix[cityOrder[secondIdx],cityOrder[(secondIdx+1)%lenOfCityOrder]] - (distance_matrix[cityOrder[secondIdx],cityOrder[firstIdx-1]] + distanceMatrix[cityOrder[firstIdx],cityOrder[(secondIdx+1)%lenOfCityOrder]])

def reverse_subarray(arr, i, j):

    if i < 0 or j >= arr.size or i > j:
        raise ValueError("Invalid indices. Ensure that 0 <= i <= j < len(arr).")
    reverseArray = arr.copy()
    reverseArray[i:j+1] = reverseArray[i:j+1][::-1]
    return reverseArray



def insert_at_index(order, index, insertion_index):
    if index < 0 or index >= order.size or insertion_index < 0 or insertion_index >= order.size:
        raise ValueError("Invalid indices. Ensure that 0 <= index, insertion_index < len(order).")
    element = order[index]
   
    new_order = np.delete(order, index)
    new_order = np.insert(new_order, insertion_index, element)
    return new_order

# Example usage
example_order = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
index = 2  # Element to move (element '3')
insertion_index = 5  # New position for the element

# Move element '3' to the new position
new_order = insert_at_index(example_order, index, insertion_index)
new_order



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







readData=pd.read_csv("Miasta29.csv",sep=";")
#readData=pd.read_csv("Dane_TSP_127.csv",sep=";")
#readData = ChangeCommaToPoint(readData)
distance_matrix = readData.iloc[:,1:].astype(int).to_numpy()
start_time = time.time()



#makeIteration(1000,distance_matrix,1000000,2050)


cityorder = np.array([20, 4, 8, 11, 5, 27, 7, 26, 23, 12, 0, 25, 28, 2, 1, 9, 3, 14, 18, 15, 22, 6, 24, 10, 21, 13, 16, 17, 19])

print(checkIfWeGetBetterRouteForInsertion(distance_matrix,cityorder,0,28))
kot = insert_at_index(cityorder,0,28)
print(kot)
print(getSumOfCities(distance_matrix,kot))
print(getSumOfCities(distance_matrix,cityorder))

print(checkIfWeGetBetterRouteForInsertion(distance_matrix,cityorder,2,11))

kot = insert_at_index(cityorder,2,11)
print(kot)
print(getSumOfCities(distance_matrix,kot))
print(getSumOfCities(distance_matrix,cityorder))


print(checkIfWeGetBetterRouteForInsertion(distance_matrix,cityorder,11,12))

kot = insert_at_index(cityorder,11,12)
print(kot)
print(getSumOfCities(distance_matrix,kot))
print(getSumOfCities(distance_matrix,cityorder))




print(checkIfWeGetBetterRouteForInsertion(distance_matrix,cityorder,0,27))

kot = insert_at_index(cityorder,0,27)
print(kot)
print(getSumOfCities(distance_matrix,kot))
print(getSumOfCities(distance_matrix,cityorder))




print(checkIfWeGetBetterRouteForInsertion(distance_matrix,cityorder,5,2))

kot = insert_at_index(cityorder,5,2)
print(kot)
print(getSumOfCities(distance_matrix,kot))
print(getSumOfCities(distance_matrix,cityorder))





print(checkIfWeGetBetterRouteForInsertion(distance_matrix,cityorder,11,4))

kot = insert_at_index(cityorder,11,4)
print(kot)
print(getSumOfCities(distance_matrix,kot))
print(getSumOfCities(distance_matrix,cityorder))


"""

print(checkIfWeGetBetterRouteForReverse(distance_matrix,cityorder,0,28))


kot = reverse_subarray(cityorder,0,28)
print(kot)
print(getSumOfCities(distance_matrix,kot))
print(getSumOfCities(distance_matrix,cityorder))

print(checkIfWeGetBetterRouteForReverse(distance_matrix,cityorder,0,27))


kot = reverse_subarray(cityorder,0,27)
print(kot)
print(getSumOfCities(distance_matrix,kot))
print(getSumOfCities(distance_matrix,cityorder))

print(checkIfWeGetBetterRouteForReverse(distance_matrix,cityorder,11,12))


kot = reverse_subarray(cityorder,11,12)
print(kot)
print(getSumOfCities(distance_matrix,kot))
print(getSumOfCities(distance_matrix,cityorder))


print(checkIfWeGetBetterRouteForReverse(distance_matrix,cityorder,4,11))

kot = reverse_subarray(cityorder,4,11)
print(kot)
print(getSumOfCities(distance_matrix,kot))
print(getSumOfCities(distance_matrix,cityorder))


print(checkIfWeGetBetterRouteForReverse(distance_matrix,cityorder,1,4))
kot = reverse_subarray(cityorder,1,4)
print(kot)


print(getSumOfCities(distance_matrix,kot))

print(getSumOfCities(distance_matrix,cityorder))
"""

"""
Test


cityorder = np.array([20, 4, 8, 11, 5, 27, 7, 26, 23, 12, 0, 25, 28, 2, 1, 9, 3, 14, 18, 15, 22, 6, 24, 10, 21, 13, 16, 17, 19])
print(calculate_route_change_for_neighbour(distance_matrix,cityorder,0,28)) expect 178 -> print(178)
print(calculate_route_change_for_neighbour(distance_matrix,cityorder,27,28)) expect 264 -> 264




"""


# The block of code to time
# [Your code here]
#[20, 4, 8, 11, 5, 27, 7, 26, 23, 12, 0, 25, 28, 2, 1, 9, 3, 14, 18, 15, 22, 6, 24, 10, 21, 13, 16, 17, 19]
#2387
# Current time after the block of code
end_time = time.time()
print(end_time-start_time)


start_time = time.time()

"""finalCitiesOrder = ClimbingAlghoritmBySwappingNotOptimal(distance_matrix,100000)
# The block of code to time
# [Your code here]

# Current time after the block of code
end_time = time.time()
print(end_time-start_time)
"""

