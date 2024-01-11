import numpy as np
import pandas as pd
import random



def save_data(best_route, distance ,method , num_iterations, filename=f"Wspinaczka_records48.txt"):
        with open(filename, 'a') as resultFile:
            resultFile.write("\n" + "=" * 25 + "\n")
            for element in best_route:
                resultFile.write(str(element+1) + ' ')
            resultFile.write(str(distance))
            resultFile.write("\n" + "Metoda: " + str(method))
            resultFile.write("\n" + "Liczba iteracji bez poprawy: " + str(num_iterations))


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
    neighbourOfFirstIndex = np.array([city(firstIdx-1),city(firstIdx+1)]) 
    neighbourOfSecondIndex = np.array([city(secondIdx-1),city(secondIdx+1)])
    return neighbourOfFirstIndex, neighbourOfSecondIndex


def checkIfWeGetBetterRoute(distanceMatrix: np.array,cityOrder: np.array,listOfNeighbour,firstIndexSwapping,SecondIndexSwapping):
    if ifSwapNeighbour(len(cityOrder)-1,firstIndexSwapping,SecondIndexSwapping) == False:
        previousRoute = checkRouteWithNeighbour(distanceMatrix,cityOrder,firstIndexSwapping,listOfNeighbour[0])+checkRouteWithNeighbour(distanceMatrix,cityOrder,SecondIndexSwapping,listOfNeighbour[1])
        newRoute = checkRouteWithNeighbour(distanceMatrix,cityOrder,firstIndexSwapping,listOfNeighbour[1])+checkRouteWithNeighbour(distanceMatrix,cityOrder,SecondIndexSwapping,listOfNeighbour[0])
        if newRoute < previousRoute:
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
    

def reverse_subarray(arr, i, j):
    reverseArray = arr.copy()
    reverseArray[i:j + 1] = reverseArray[i:j + 1][::-1]
    return reverseArray

def insert_at_index(order, index, insertion_index):
    element = order[index]
    new_order = np.delete(order, index)
    new_order = np.insert(new_order, insertion_index, element)
    return new_order


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

    length_before = (distanceMatrix[cityOrder[i-1], cityOrder[i]] +
                     distanceMatrix[cityOrder[i+1], cityOrder[(i+2)%lenght]])

    length_after = (distanceMatrix[cityOrder[i-1], cityOrder[i+1]] +
                    distanceMatrix[cityOrder[i], cityOrder[(i+2)%lenght]])

    return length_after - length_before

def calculateRouteChangeForInsertion(distanceMatrix: np.array,cityOrder: np.array,indexOfCityInCityOrder,indexToInsert):
    lenOfCityOrder = len(cityOrder)
    if ifSwapNeighbour(lenOfCityOrder-1,indexOfCityInCityOrder,indexToInsert) == False:
        if indexOfCityInCityOrder < indexToInsert:
            lengthBefore = distanceMatrix[cityOrder[indexOfCityInCityOrder],cityOrder[indexOfCityInCityOrder-1]] + distanceMatrix[cityOrder[indexOfCityInCityOrder],cityOrder[(indexOfCityInCityOrder+1)%lenOfCityOrder]] + distanceMatrix[cityOrder[indexToInsert],cityOrder[(indexToInsert+1)%lenOfCityOrder]] 

            lengthAfter = distanceMatrix[cityOrder[indexOfCityInCityOrder],cityOrder[indexToInsert]] + distanceMatrix[cityOrder[indexOfCityInCityOrder],cityOrder[(indexToInsert+1)%lenOfCityOrder]] + distanceMatrix[cityOrder[indexOfCityInCityOrder+1],cityOrder[(indexOfCityInCityOrder-1)]] 
            return lengthAfter - lengthBefore
        lengthBefore = (distanceMatrix[cityOrder[indexOfCityInCityOrder],cityOrder[indexOfCityInCityOrder-1]] + 
                        distanceMatrix[cityOrder[indexOfCityInCityOrder],cityOrder[(indexOfCityInCityOrder+1)%lenOfCityOrder]] + 
                        distanceMatrix[cityOrder[indexToInsert],cityOrder[indexToInsert-1]] )
        lengthAfter = (distanceMatrix[cityOrder[indexOfCityInCityOrder],cityOrder[indexToInsert]] +
                        distanceMatrix[cityOrder[indexOfCityInCityOrder],cityOrder[indexToInsert-1]] +
                        distanceMatrix[cityOrder[(indexOfCityInCityOrder+1)%lenOfCityOrder],cityOrder[(indexOfCityInCityOrder-1)]]) 
        return lengthAfter - lengthBefore
    elif abs(indexOfCityInCityOrder-indexToInsert) == 1:
        firstIdx = indexOfCityInCityOrder
        secIdx = indexToInsert
        if indexToInsert < indexOfCityInCityOrder:
            firstIdx = indexToInsert
            secIdx = indexOfCityInCityOrder
        return calculate_route_change_for_neighbour(distanceMatrix,cityOrder,firstIdx,secIdx)
    return 0
 

def calculateRouteChangeForReverse(distanceMatrix: np.array,cityOrder: np.array,firstIdx,secondIdx):
    lenOfCityOrder = len(cityOrder)
    if firstIdx==0 and secondIdx == lenOfCityOrder-1:
        return 0
    if firstIdx - secondIdx == -1:
        return calculate_route_change_for_neighbour(distanceMatrix,cityOrder,firstIdx,secondIdx)
    return  (distance_matrix[cityOrder[secondIdx],cityOrder[firstIdx-1]] 
             + distanceMatrix[cityOrder[firstIdx],cityOrder[(secondIdx+1)%lenOfCityOrder]]) -(
            distance_matrix[cityOrder[firstIdx],cityOrder[firstIdx-1]] 
             + distance_matrix[cityOrder[secondIdx],cityOrder[(secondIdx+1)%lenOfCityOrder]])

def checkIfWeGetBetterRouteForReverse(distanceMatrix: np.array,cityOrder: np.array,firstIndex,SecondIndex):
    distanceChange = calculateRouteChangeForReverse(distanceMatrix,cityOrder,firstIndex,SecondIndex)
    if distanceChange < 0:
        return reverse_subarray(cityOrder, firstIndex, SecondIndex),True
    return cityOrder,False

def checkIfWeGetBetterRouteForInsertion(distanceMatrix: np.array,cityOrder: np.array,firstIndex,SecondIndex):
    distanceChange = calculateRouteChangeForInsertion(distanceMatrix,cityOrder,firstIndex,SecondIndex)
    if distanceChange < 0:
        return insert_at_index(cityOrder, firstIndex, SecondIndex),True
    return cityOrder,False

  
def getSumOfCities(distanceMatrix,cityOrders):
    sum = 0
    for i in range(-1,len(cityOrders)-1):
        sum = sum + distanceMatrix[cityOrders[i],cityOrders[i+1]]
    return sum


def ClimbingAlghoritm(distanceMatrix,cityOrder,howManyIterationWithoutImprovement,cities, method="reverse", howManyIteration = None ):

    i = 0
    k = 0
    iteration = howManyIterationWithoutImprovement
    if howManyIteration is not None:
        iteration = howManyIteration


    while i < iteration:
       

        indexOfTable = random.sample(cities, 2)
        changeCityOrder = None
        if method == "swapping":
            getNeighbourCities = GetCities(cityOrder,indexOfTable[0],indexOfTable[1])
            changeCityOrder = checkIfWeGetBetterRoute(distanceMatrix,cityOrder,getNeighbourCities,indexOfTable[0],indexOfTable[1])
        elif method == "reverse":
            if (indexOfTable[0] > indexOfTable[1]):
                   indexOfTable[0], indexOfTable[1] = indexOfTable[1], indexOfTable[0]
            changeCityOrder = checkIfWeGetBetterRouteForReverse(distanceMatrix,cityOrder,indexOfTable[0],indexOfTable[1])
        else:
            if (indexOfTable[0] > indexOfTable[1]):
                indexOfTable[0], indexOfTable[1] = indexOfTable[1], indexOfTable[0]
            changeCityOrder = checkIfWeGetBetterRouteForInsertion(distanceMatrix,cityOrder,indexOfTable[0],indexOfTable[1])
        cityOrder = changeCityOrder[0]
        i = i+1
        k= k+1
        if changeCityOrder[1] == True and howManyIteration is None:
            i = 0
    return cityOrder 


def makeIteration(repetition,distanceMatrix,howManyIterationWithoutImprovem,acceptanceValue,method="reverse", howManyIteration = None):
    for i in range (repetition):
        cityOrder = getRandomRouteCities(distanceMatrix.shape[0])
        cities = range(len(cityOrder))
        finalCitiesOrder = ClimbingAlghoritm(distanceMatrix,cityOrder,howManyIterationWithoutImprovem,cities,method,howManyIteration)
        sumOfFinalResult = getSumOfCities(distanceMatrix,finalCitiesOrder)
        print(sumOfFinalResult)
        print(finalCitiesOrder)

        if sumOfFinalResult < acceptanceValue:
            save_data(finalCitiesOrder,sumOfFinalResult,method,howManyIterationWithoutImprovem)




#readData=pd.read_csv("Miasta29.csv",sep=";", decimal=",")
readData=pd.read_csv("Dane_TSP_48.csv", sep=";", decimal=",")
readData=pd.read_csv("Dane_TSP_76.csv", sep=";", decimal=",")
readData=pd.read_csv("Dane_TSP_127.csv", sep=";", decimal=",")
distance_matrix = readData.iloc[:,1:].astype(float).to_numpy()
makeIteration(1,distance_matrix, 2200, 2000, method="reverse", howManyIteration = 1000000)