import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum
import random
import csv

class InitializationMethod(Enum):
    RANDOM = "random"
    HE = "he"
    XAVIER_GLOROT = "xavier_glorot"

class TrainingSetSelection(Enum):
    RANDOM = "random" # dobór w pełni losowy
    STRATIFIEDSAMPLING = "stratified_sampling" #dobór losowy, ale z zachowaniem proporcji danych (odpowiedzi)
    BOOTSTRAPPING = "bootstrapping" #dobór losowy, ale z możliwością wielokrotnego wyboru tych samych danych
    RANDOMWITHIMPORTANCE = "random_with_importance" #dobór w pełni losowy, po którym następnie powtórzenie najmniej licznych danych (odpowiedzi) tyle razy, żeby wyrównać wszystkie zbiory 

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

def add_random_data(how_much_data_add, index_of_col, data, labels):
    indeksy = np.where(labels[:, index_of_col] == 1)[0]
    wanted_data = data[indeksy]
    wanted_labels = labels[indeksy]
    num_of_labels = np.shape(wanted_labels)[0]
    
    new_data = data.copy()  # Tworzenie kopii danych
    new_labels = labels.copy()  # Tworzenie kopii etykiet

    for _ in range(how_much_data_add):
        index_of_adding_row = random.randint(0, (num_of_labels-1))
        new_data = np.vstack([new_data, wanted_data[index_of_adding_row]])
        new_labels = np.vstack([new_labels, wanted_labels[index_of_adding_row]])

    return new_data, new_labels

# Dzielenie danych na zbior uczacy i testowy
def get_train_data_and_test_data(data,labels,test_sample_percent,type_of_split):
    data_length = data.shape[0]
    if TrainingSetSelection.RANDOM == type_of_split:
        indices_for_test =  random.sample(range(0, data_length), int(test_sample_percent*data_length))
        indices_for_train = [x for x in range(data_length) if x not in indices_for_test]
        returner1 = extend_array(labels[indices_for_train])
        returner2 = extend_array(labels[indices_for_test])
        return data[indices_for_train,:], returner1, data[indices_for_test,:], returner2
    elif TrainingSetSelection.STRATIFIEDSAMPLING == type_of_split:
        return 0
    elif TrainingSetSelection.BOOTSTRAPPING == type_of_split:
        test_sample_size = int(test_sample_percent * data_length)
        unique_numbers = set()
        random_numbers = []
        while len(unique_numbers) < test_sample_size:
            new_number = random.randint(0, data_length - 1)
            if new_number not in unique_numbers:
                unique_numbers.add(new_number)
            random_numbers.append(new_number)
        returner1 = extend_array(labels[random_numbers])
        returner2 = extend_array(labels[random_numbers])
        print(str(np.shape(data[random_numbers,:])))
        print(str(np.shape(data[random_numbers,:])))
        return data[random_numbers,:], returner1, data[random_numbers,:], returner2
    elif TrainingSetSelection.RANDOMWITHIMPORTANCE == type_of_split:
        train_data, train_label, test_data, test_label = get_train_data_and_test_data(data,labels,test_sample_percent,type_of_split = TrainingSetSelection.RANDOM)
        numbers_of_datas = list(range(10))
        for i in range(0,10):
            numbers_of_datas[i] = np.count_nonzero(train_label[:, i] == 1)
        for i in range(0,10):
            train_data, train_label = add_random_data(max(numbers_of_datas) - numbers_of_datas[i], i, train_data, train_label)
        return train_data, train_label, test_data, test_label
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
def relu(neurons):
    A = np.maximum(0, neurons)
    activation_history = neurons
    return A, activation_history

def relu_backward(dA, activation_history):
    dZ = np.copy(dA)
    dZ[activation_history <= 0] = 0
    return dZ

def softmax(neurons):
    e_x = np.exp(neurons - np.max(neurons))
    A = e_x / np.sum(e_x, axis=0, keepdims=True)
    return A, neurons

def softmax_backward(dA, activation_history):
    s, _ = softmax(activation_history)
    dZ = np.zeros_like(s)
    
    for i in range(dA.shape[1]):
        dZ[:, i] = (np.diag(s[:, i]) @ dA[:, i]) - np.outer(s[:, i], s[:, i]) @ dA[:, i]
    
    return dZ


# Inicjalizacja parametrów sieci neuronowej
def initialize_parameters_deep(layer_dims):
    parameters = {}
    length_of_layers = len(layer_dims)
    
    for l in range(1, length_of_layers):
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
    neurons = W @ A + b
    activation_history = (A, W, b)
    return neurons, activation_history

def linear_activation_forward(A_prev, W, b, activation): 
    """activation na enum i zmienić mu nazwę. można dodać więcej funkcji aktywacji"""
    if activation == "relu":
        neurons, linear_forward_history = linear_forward(A_prev, W, b)
        A, activation_history = relu(neurons)
    elif activation == "softmax":
        neurons, linear_forward_history = linear_forward(A_prev, W, b)
        A, activation_history = softmax(neurons)
    activations_history = (linear_forward_history, activation_history)
    return A, activations_history

# Obliczanie entropii krzyżowej
def compute_cost(model_results, Y):
    number_of_data = Y.shape[1]
    cost = (-1 / number_of_data) * np.sum(Y * np.log(model_results + 1e-4) + (1 - Y) * np.log(1 - model_results + 1e-4))
    return cost

# Wsteczna propagacja przez warstwy
def linear_backward(dZ, activation_history):
    A_prev, W, b = activation_history
    number_of_data = A_prev.shape[1]
    dW = (1 / number_of_data) * (dZ @ A_prev.T)
    db = (1 / number_of_data) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = W.T @ dZ
    return dA_prev, dW, db

def linear_activation_backward(dA, activations_history, activation):
    """activation na enum i zmienić mu nazwę. można dodać więcej funkcji aktywacji"""
    linear_cache, activation_history = activations_history
    if activation == "relu":
        dZ = relu_backward(dA, activation_history)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == "softmax":
        dZ = softmax_backward(dA, activation_history)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db

def backward_propagation(actual_layers, Y,  activations_history):
    gradient = {}
    L = len(activations_history)
    dAL = - ((Y / (actual_layers + 1e-4)) - ((1 - Y) / (1 - actual_layers + 1e-4)))
    number_of_layers=len(layers_dims)
    current_cache =  activations_history[number_of_layers-2]
    gradient["dA"+str(number_of_layers-1)], gradient["dW"+str(number_of_layers-1)], gradient["db"+str(number_of_layers-1)] = linear_activation_backward(dAL, current_cache, activation = "softmax")
    for l in reversed(range(L-1)):
        current_cache =  activations_history[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(gradient["dA" + str(l + 2)], current_cache, activation = "relu")
        gradient["dA" + str(l + 1)] = dA_prev_temp
        gradient["dW" + str(l + 1)] = dW_temp
        gradient["db" + str(l + 1)] = db_temp
    return gradient

def update_parameters(parameters, grads, learning_rate):
    L = len(layers_dims) - 1
    for l in range(L):
        #print("Oto sprawdzane LLLLLLLLLLLLLLLLL"+str(l))
        #print("W "+str(learning_rate * grads["dW" + str(l + 1)]))
        #print("b "+str(learning_rate * grads["db" + str(l + 1)]))
        parameters["W" + str(l + 1)] -= learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] -= learning_rate * grads["db" + str(l + 1)]
    return parameters

# Testowanie wytrenowanego modelu na danych testowych
def check_test(X, params):
    actual_layers,  activations_history = forward_propagation(X, params)
    return actual_layers,  activations_history


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
    actual_layers -- last post-activation value
    activations_history -- list of  activations_history containing every activation_history of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
    """

    activations_history = []
    A = X
    number_of_layers = len(parameters) // 2                  # number of layers in the neural network

    # Implement [LINEAR -> RELU]*(L-1). Add "activation_history" to the " activations_history" list.
    for l in range(1, number_of_layers):
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
        #Z = (W @ A_prev) + b
        #A = relu(Z)
        activations_history.append(activation_history)

    # Implement LINEAR -> SOFTMAX. Add "activation_history" to the "activations_history" list.
    W = parameters['W' + str(number_of_layers)]
    b = parameters['b' + str(number_of_layers)]
    actual_layers, activation_history = linear_activation_forward(A,W,b,"softmax")
    activations_history.append(activation_history)
    
    return actual_layers, activations_history

# Tłumaczy macierz prawdopodobieństw na odpowiedzi
def translate_matrix_of_probabilities_to_matrix_of_answers(array):
    # Kopiowanie oryginalnej tablicy numpy
    result = array.copy()
    
    # Iteracja po każdym wierszu tablicy numpy
    for row in range(array.shape[0]):
        # Znajdowanie indeksu maksymalnej wartości w danym wierszu
        max_index = np.argmax(array[row])
        
        # Ustawienie 1 w komórce o maksymalnej wartości w danym wierszu
        result[row] = 0
        result[row, max_index] = 1
    
    return result

# Porównuje dwie macierze i zwraca średnią zgodność wierszy
def matrix_comparison(arr1, arr2):
    if arr1.shape != arr2.shape:
        raise ValueError("Tablice mają różne kształty. Porównanie niemożliwe.")

    rows = arr1.shape[0]
    columns = arr1.shape[1]

    returner = 0  # Zmienna lokalna do zliczania identycznych wierszy

    for i in range(rows):
        row_match = True
        for j in range(columns):
            if arr1[i][j] != arr2[i][j]:
                row_match = False
                break

        if row_match:
            returner += 1

    return returner / rows if rows > 0 else 0

def neural_network(X, Y, layers_dims, learning_rate, epoka):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SOFTMAX.
    
    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    layers_dims -- dimensions of the layers (n_x, n_h, n_y)
    learning_rate -- learning rate of the gradient descent update rule
    epoka -- number of iterations of the optimization loop
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    costs = []                         # keep track of cost
    
    # Parameters initialization
    parameters = initialize_parameters_deep(layers_dims)

    # Loop (gradient descent)
    for i in range(0, epoka):

        # Forward propagation
        actual_layers,  activations_history = forward_propagation(X, parameters)
        #print(np.shape(Y))

        # Compute cost
        cost = compute_cost(actual_layers, Y)

        # Backward propagation
        grads = backward_propagation(actual_layers, Y,  activations_history)

        # Update parameters
        parameters = update_parameters(parameters, grads, learning_rate)
        
        # Print the cost every 100 training example
        if i % 10 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
            costs.append(cost)
            
    return parameters



# Wczytanie danych treningowych i testowych
mnist_data_csv_1 = pd.read_csv("mnist1.csv", sep = ",")
mnist_data_csv_2 = pd.read_csv("mnist2.csv", sep = ",")

# Podział danych na etykiety i piksele
mnist_labels_1 = np.array(mnist_data_csv_1.iloc[:, 0])
mnist_labels_2 = np.array(mnist_data_csv_2.iloc[:, 0])
mnist_data_1 = np.array(mnist_data_csv_1.iloc[:, 1:], dtype='int16')
mnist_data_2 = np.array(mnist_data_csv_2.iloc[:, 1:], dtype='int16')

# Łączenie danych do jednego arrayu
all_mnist_labels = np.concatenate((mnist_labels_1,mnist_labels_2),axis=0)
all_data = np.concatenate((mnist_data_1,mnist_data_2),axis=0)

# Dzielenie danych na zbiór uczący i testowy
percent_of_test_data = 0.1
list_of_datas = get_train_data_and_test_data(all_data,all_mnist_labels,percent_of_test_data,TrainingSetSelection.RANDOM)
train_data = np.transpose(list_of_datas[0])
train_label = np.transpose(list_of_datas[1])
test_data = np.transpose(list_of_datas[2])
test_label = np.transpose(list_of_datas[3])


# Do każdego debila który będzie to zmieniał. PIERWSZA I OSTATNIA LICZBA NIE MA PRAWA SIĘ ZMIENIĆ !!!!!
layers_dims = [784, 392, 196, 98, 49, 10] # Do każdego debila który będzie to zmieniał. PIERWSZA I OSTATNIA LICZBA NIE MA PRAWA SIĘ ZMIENIĆ !!!!!
# Do każdego debila który będzie to zmieniał. PIERWSZA I OSTATNIA LICZBA NIE MA PRAWA SIĘ ZMIENIĆ !!!!!

parameters = neural_network(train_data, train_label, layers_dims, learning_rate=0.002, epoka=250)
predictions, _ = check_test(test_data, parameters)
#print(predictions)
print("Macierz odpowiedzi ma rozmiary: " + str(np.shape(predictions)))
print(str(np.max(predictions)))
predictions = translate_matrix_of_probabilities_to_matrix_of_answers(np.transpose(predictions))
print(str(matrix_comparison(predictions,np.transpose(test_label))))
save_array_as_csv(predictions,'Answers.csv')



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