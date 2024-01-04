import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random



def get_train_data_and_test_data(data,labels,test_sample_percent):
    data_length = data.shape[0]
    indices_for_test =  random.sample(range(0, data_length), int(test_sample_percent*data_length))
    indices_for_train = [x for x in range(data_length) if x not in indices_for_test]
    return data[indices_for_train,:], labels[indices_for_train], data[indices_for_test,:], labels[indices_for_test]



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
    cache = Z
    return A, cache

def softmax_backward(dA, cache):
    Z = cache
    s, _ = softmax(Z)
    dZ = np.zeros_like(s)
    
    for i in range(dA.shape[1]):
        dZ[:, i] = np.dot(np.diag(s[:, i]), dA[:, i]) - np.outer(s[:, i], s[:, i]) @ dA[:, i]
    
    return dZ

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

def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation="relu")
        caches.append(cache)
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation="softmax")
    caches.append(cache)
    return AL, caches

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
    current_cache = caches[L - 1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation="softmax")
    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, activation="relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
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
        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        grads = L_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)
        cost_plot[i] = cost

    if print_cost:
        plot_graph(cost_plot)
    return parameters

# Testowanie wytrenowanego modelu na danych testowych
def check_test(X, params):
    AL, caches = L_model_forward(X, params)
    return AL, caches


# Wczytanie danych treningowych i testowych
mnist_data_csv_1 = pd.read_csv("mnist1.csv", sep = ",")
mnist_data_csv_2 = pd.read_csv("mnist2.csv", sep = ",")

# Podział danych na etykiety i piksele
mnist_labels_1 = np.array(mnist_data_csv_1.loc[:, 'label'])
mnist_labels_2 = np.array(mnist_data_csv_2.loc[:, 'label'])
mnist_data_1 = np.array(mnist_data_csv_1.loc[:, mnist_data_csv_1.columns != 'label'])
mnist_data_2 = np.array(mnist_data_csv_2.loc[:, mnist_data_csv_2.columns != 'label'])


all_mnist_labels = np.concatenate((mnist_labels_1,mnist_labels_2),axis=0)
all_data = np.concatenate((mnist_data_1,mnist_data_2),axis=0)

train_data = get_train_data_and_test_data(all_data,all_mnist_labels,0.1)[0]
train_label = get_train_data_and_test_data(all_data,all_mnist_labels,0.1)[1]


test_data = get_train_data_and_test_data(all_data,all_mnist_labels,0.1)[2]
test_label = get_train_data_and_test_data(all_data,all_mnist_labels,0.1)[3]



# Trenowanie modelu
layers_dims = [784, 700, 600, 500, 400, 300, 200, 100, 50, 10]
parameters1 = L_layer_model(train_data, train_label, layers_dims, learning_rate=0.0005, num_iterations=22, print_cost=True)
print("Trenowanie zakończone")



predictions, _ = check_test(test_data, parameters1)

print(predictions)

"""



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