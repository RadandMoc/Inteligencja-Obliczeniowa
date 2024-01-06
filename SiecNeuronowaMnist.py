import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from enum import Enum
import random
import csv

# Funkcja przekształca array tworząc macierz odpowiedzi
def extend_array(array):
    # Sprawdzenie czy w kolumnie znajdują się wartości od 0 do 9
    unikalne_wartosci = np.unique(array)
    if not np.array_equal(unikalne_wartosci, np.arange(10)):
        raise ValueError("Kolumna powinna zawierać wartości od 0 do 9")

    # Tworzenie nowego arraya z zerami o wymiarach: liczba wierszy x 10 kolumn
    result = np.zeros((array.shape[0], 10), dtype=int)

    # Indeksowanie wartości 1 na podstawie wartości w kolumnie oryginalnego arraya
    for i, val in enumerate(array.flatten()):
        result[i, val] = 1

    return result

# Dzielenie danych na zbior uczacy i testowy
def get_train_data_and_test_data(data,labels,test_sample_percent,want_random_order):
    data_length = data.shape[0]
    if want_random_order:
        indices_for_test =  random.sample(range(0, data_length), int(test_sample_percent*data_length))
        indices_for_train = [x for x in range(data_length) if x not in indices_for_test]
        returner1 = extend_array(labels[indices_for_train])
        returner2 = extend_array(labels[indices_for_test])
        return data[indices_for_train,:], returner1, data[indices_for_test,:], returner2
    else:
        split_index = int((1-test_sample_percent) * data_length)
        returner1 = extend_array(labels[:split_index])
        returner2 = extend_array(labels[split_index:])
        return data[:split_index, :], returner1, data[split_index:, :], returner2

def save_array_as_csv(array, file_path):
    # Jeśli tablica jest jednowymiarowa, przekształć ją w tablicę 2D (jeden wiersz)
    if array.ndim == 1:
        array = array.reshape(1, -1)

    try:
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(array)
        print(f"Plik CSV został pomyślnie zapisany jako {file_path}")
    except Exception as e:
        print(f"Błąd podczas zapisywania pliku CSV: {e}")

# Definicja funkcji aktywacji i ich pochodnych
def relu(Z):
    A = np.maximum(0, Z)
    cache = Z
    return A, cache

def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    assert (dZ.shape == Z.shape)
    return dZ

def softmax(Z):
    e_x = np.exp(Z - np.max(Z))
    A = e_x / np.sum(e_x, axis=0, keepdims=True)
    return A, Z

def softmax_backward(dA, cache):
    Z = cache
    s, _ = softmax(Z)
    dZ = np.zeros_like(s)
    
    for i in range(dA.shape[1]):
        dZ[:, i] = np.dot(np.diag(s[:, i]), dA[:, i]) - np.outer(s[:, i], s[:, i]) @ dA[:, i]
    
    return dZ

def sigmoid(Z):
    return 1/(1+np.exp(-Z))



# Inicjalizacja parametrów sieci neuronowej
def initialize_parameters_deep(layer_dims):
    parameters = {}
    L = len(layer_dims)
    
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
    
    return parameters

def linear_forward(A, W, b):
    """

    Przekazywanie przez warstwy w przód
    
    Parametry:
    
    A - aktywacja poprzedniej warstwy
    
    W - wagi
    
    b - bias
    
    """
    #print("W shape="+str(np.shape(W)))
    #print("A_prev shape="+str(np.shape(A)))
    #print("b shape ="+str(np.shape(b)))
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    assert Z.shape == (W.shape[0], A.shape[1])
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    if activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    elif activation == "softmax":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = softmax(Z)
    cache = (linear_cache, activation_cache)
    return A, cache

# Obliczanie kosztu
def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = (-1 / m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL)))
    return cost

# Wsteczna propagacja przez warstwy
def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == "softmax":
        dZ = softmax_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    M=len(layers_dims)
    current_cache = caches[M-2]
    grads["dA"+str(M-1)], grads["dW"+str(M-1)], grads["db"+str(M-1)] = linear_activation_backward(dAL, current_cache, activation = "softmax")
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, activation = "relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads

def update_parameters(parameters, grads, learning_rate):
    L = len(layers_dims) - 1
    for l in range(L):
        #print("Oto sprawdzane LLLLLLLLLLLLLLLLL"+str(l))
        #print("W "+str(learning_rate * grads["dW" + str(l + 1)]))
        #print("b "+str(learning_rate * grads["db" + str(l + 1)]))
        parameters["W" + str(l + 1)] -= learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] -= learning_rate * grads["db" + str(l + 1)]
    return parameters

# Funkcja rysująca wykres funkcji kosztu
def plot_graph(cost_plot):
    x_values = list(range(1, len(cost_plot) + 1))
    plt.xlabel('Iteracja')
    plt.ylabel('Koszt')
    plt.plot(x_values, cost_plot, color='green')
    plt.show()

# Model sieci neuronowej L-warstwowej
def L_layer_model(X, Y, layers_dims, learning_rate, num_iterations, print_cost=False):
    costs = []
    cost_plot = np.zeros(num_iterations)
    parameters = initialize_parameters_deep(layers_dims)
    for i in range(num_iterations):
        AL, caches = forward_propagation(X, parameters)
        cost = compute_cost(AL, Y)
        grads = L_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)
        cost_plot[i] = cost

    if print_cost:
        plot_graph(cost_plot)
    return parameters

# Testowanie wytrenowanego modelu na danych testowych
def check_test(X, params):
    AL, caches = forward_propagation(X, params)
    return AL, caches


class InitializationMethod(Enum):
    RANDOM = "random"
    HE = "he"
    XAVIER_GLOROT = "xavier_glorot"



def initialize_parameters(layers_dims, method=InitializationMethod.RANDOM):
    """
    Inicjalizuje wagi i biasy dla każdej warstwy w sieci neuronowej zgodnie z wybraną metodą.

    Argumenty:
    layers_dims -- lista zawierająca liczbę neuronów w każdej warstwie.
    method -- metoda inicjalizacji (InitializationMethod).

    Zwraca:
    parameters -- słownik zawierający parametry "W0", "b0", ..., "WL", "bL".
    """

    np.random.seed(3)  # Ustawienie ziarna dla spójności wyników
    parameters = {}
    L = len(layers_dims)  # liczba warstw w sieci

    for l in range(1, L):
        if method == InitializationMethod.HE:
            parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * np.sqrt(2 / layers_dims[l-1])
        elif method == InitializationMethod.XAVIER_GLOROT:
            limit = np.sqrt(6 / (layers_dims[l-1] + layers_dims[l]))
            parameters['W' + str(l)] = np.random.uniform(-limit, limit, (layers_dims[l], layers_dims[l-1]))
        else:  # DEFAULT: Random initialization
            parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * 0.01

        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

    return parameters


def forward_propagation(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation.
    
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    
    Returns:
    AL -- last post-activation value
    caches -- list of caches containing every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
    """

    activations_history = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network

    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A 
        W = parameters['W' + str(l)]
        b = parameters['b' + str(l)]
        #save_array_as_csv(W,'zmiennaW.csv')
        #save_array_as_csv(A_prev,'zmiennaA.csv')
        #save_array_as_csv(b,'zmiennaB.csv')
        #print('W' + str(l) +" shape="+str(np.shape(W)))
        #print("A_prev shape="+str(np.shape(A_prev)))
        #print("b shape ="+str(np.shape(b)))
        A, activation_history = linear_activation_forward(A_prev, W, b, "relu")
        #Z = np.dot(W, A_prev) + b
        #A = relu(Z)
        activations_history.append(activation_history)

    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    W = parameters['W' + str(L)]
    b = parameters['b' + str(L)]
    AL, activation_history = linear_activation_forward(A,W,b,"softmax")
    activations_history.append(activation_history)
    
    return AL, activations_history



def neural_network(X, Y, layers_dims, learning_rate, num_iterations):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SOFTMAX.
    
    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    layers_dims -- dimensions of the layers (n_x, n_h, n_y)
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    costs = []                         # keep track of cost
    
    # Parameters initialization
    parameters = initialize_parameters_deep(layers_dims)

    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation
        AL, caches = forward_propagation(X, parameters)
        #print(np.shape(Y))

        # Compute cost
        cost = compute_cost(AL, Y)

        # Backward propagation
        grads = L_model_backward(AL, Y, caches)

        # Update parameters
        parameters = update_parameters(parameters, grads, learning_rate)
        
        # Print the cost every 100 training example
        if i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
            costs.append(cost)
            
    return parameters



# Wczytanie danych treningowych i testowych
mnist_data_csv_1 = pd.read_csv("mnist1.csv", sep = ",")
mnist_data_csv_2 = pd.read_csv("mnist2.csv", sep = ",")

# Podział danych na etykiety i piksele
mnist_labels_1 = np.array(mnist_data_csv_1.loc[:, 'label'])
mnist_labels_2 = np.array(mnist_data_csv_2.loc[:, 'label'])
mnist_data_1 = np.array(mnist_data_csv_1.loc[:, mnist_data_csv_1.columns != 'label'], dtype = 'int16')
mnist_data_2 = np.array(mnist_data_csv_2.loc[:, mnist_data_csv_2.columns != 'label'], dtype = 'int16')

# Łączenie danych do jednego arrayu
all_mnist_labels = np.concatenate((mnist_labels_1,mnist_labels_2),axis=0)
all_data = np.concatenate((mnist_data_1,mnist_data_2),axis=0)

# Dzielenie danych na zbiór uczący i testowy
percent_of_test_data = 0.4
list_of_datas = get_train_data_and_test_data(all_data,all_mnist_labels,percent_of_test_data,True)
train_data = np.transpose(list_of_datas[0])
train_label = np.transpose(list_of_datas[1])
test_data = np.transpose(list_of_datas[2])
test_label = np.transpose(list_of_datas[3])

print("train_data shape="+str(np.shape(train_data)))
print("train_label shape="+str(np.shape(train_label)))
print("test shape ="+str(np.shape(test_data)))
print("test_label shape="+str(np.shape(test_label)))

# Sprawdzenie danych zbiorów testowych i uczących w formie zapisu do plikow csv celem latwiejszej inspekcji
#save_array_as_csv(train_data,'DaneUczace.csv')
#save_array_as_csv(train_label,'ZnakiUczace.csv')
#save_array_as_csv(test_data,'DaneTestowe.csv')
#save_array_as_csv(test_label,'ZnakiTestowe.csv')
#save_array_as_csv(all_mnist_labels,'Znaki.csv')
#save_array_as_csv(all_data,'Dane.csv')





layers_dims = [784, 700, 600, 500, 400, 300, 200, 100, 50, 10]
parameters = neural_network(train_data, train_label, layers_dims, learning_rate=0.0005, num_iterations=22)
predictions, _ = check_test(test_data, parameters)
#print(predictions)
#print(str(np.shape(predictions)))
print(str(np.max(predictions)))
save_array_as_csv(np.transpose(predictions),'Answers.csv')



# Trenowanie modelu
#layers_dims = [784, 700, 600, 500, 400, 300, 200, 100, 50, 10]
"""parameters1 = L_layer_model(train_data, train_label, layers_dims, learning_rate=0.0005, num_iterations=22, print_cost=True)
print("Trenowanie zakończone")



predictions, _ = check_test(test_data, parameters1)

print(predictions)

print(str(np.max(predictions)))
save_array_as_csv(np.transpose(predictions),'Answers.csv')



results = []

# Tworzenie wyników na podstawie predykcji
for i in range(28000):
    tmp = {}
    for j in range(10):
        tmp[j] = predictions[j, i]
    max_key = max(tmp, key=tmp.get)
    results.append(max_key)

# Wyświetlenie histogramu wyników testowych
plt.hist(results, edgecolor='black', linewidth=1.2)
plt.title("Histogram danych testowych")
plt.axis([0, 9, 0, 5000])
plt.xlabel("Cyfra")
plt.ylabel("Wystąpienia w zestawie testowym")
plt.show()

# Zapisanie wyników do pliku submission.csv
np.savetxt('submission.csv', np.c_[range(1, 28001), results], delimiter=',', header='ImageId,Label', comments='', fmt='%d')
print("Testowanie zakończone")
"""