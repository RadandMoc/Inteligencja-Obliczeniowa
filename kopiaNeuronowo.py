import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum
import random
import sys
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

class ActivationFunction(Enum):
    Relu = "relu"
    Softmax = "softmax"
    Tanh = "tanh"
    Sigmoid = "sigmoid"




def batch_normalization(X, gamma, beta, epsilon=1e-5):
    mean = np.mean(X, axis=0)
    variance = np.var(X, axis=0)

    X_normalized = (X - mean) / np.sqrt(variance + epsilon)
    out = gamma * X_normalized + beta

    return out


def get_train_data_and_test_data(data,labels,test_sample_percent,type_of_split):
    np.random.seed(1)
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




def relu(neurons):
    return np.maximum(0, neurons)

def softmax(neurons):
    e_x = np.exp(neurons - np.max(neurons, axis=0, keepdims=True))
    A = e_x / np.sum(e_x, axis=0, keepdims=True)
    return A

def sigmoid(neurons):
    A = 1 / (1 + np.exp(-neurons))
    return A

def sigmoid_backward(dA, Z):
    s = sigmoid(Z)
    dZ = dA * s * (1 - s)
    return dZ

def tanh(neurons):
    A = np.tanh(neurons)
    return A

def tanh_backward(dA, Z):
    t = tanh(Z)
    dZ = dA * (1 - np.square(t))
    return dZ

def relu_backward(dA, Z):
    s = relu(Z)
    mask = s > 0
    dZ = dA * mask
    return dZ


def softmax_backward(dA, Z):
    s = softmax(Z)
    dZ = s - dA  # Prosta różnica pomiędzy aktywacją softmax a prawdziwymi etykietami
    return dZ


def initialize_parameters(layers_dims, method=InitializationMethod.RANDOM):
    """
    Inicjalizuje wagi i biasy dla każdej warstwy w sieci neuronowej zgodnie z wybraną metodą.

    Argumenty:
    layers_dims -- lista zawierająca liczbę neuronów w każdej warstwie.
    method -- metoda inicjalizacji (InitializationMethod).

    Zwraca:
    parameters -- lista słowników, gdzie każdy słownik zawiera wagi i biasy dla jednej warstwy.
    """

    np.random.seed(3)
    weights = []
    biases = [] 

    for l in range(1, len(layers_dims)):
        weight = 0
        if method == InitializationMethod.HE:
            weight = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(2 / layers_dims[l - 1])
        elif method == InitializationMethod.XAVIER_GLOROT:
            limit = np.sqrt(6 / (layers_dims[l - 1] + layers_dims[l]))
            weight = np.random.uniform(-limit, limit, (layers_dims[l], layers_dims[l - 1]))
        else:  # DEFAULT: Random initialization
            weight = np.random.randn(layers_dims[l], layers_dims[l - 1]) * 0.01

        bias = np.zeros((layers_dims[l], 1))
        weights.append(weight)
        biases.append(bias)
    return weights,biases


def activation_function_forward(A_prev, weights, bias, activation):
    """
    Implement forward activation for a single network layer.

    Arguments:
    A_prev -- activations from previous layer (or input data)
    weights -- weights matrix for this layer
    bias -- bias vector for this layer
    activation -- activation function to be used (from ActivationFunction enum)

    Returns:
    A -- output of the activation function for this layer
    Z -- linear component (weighted input) for this layer
    """
    # Obliczanie liniowej części (Z = W * A_prev + b)
    Z = weights @ A_prev + bias
    A = 0
    # Aplikacja funkcji aktywacji
    if activation == ActivationFunction.Relu:
        A = relu(Z)
    elif activation == ActivationFunction.Softmax:
        A = softmax(Z)
    elif activation == ActivationFunction.Sigmoid:
        A = sigmoid(Z)
    elif activation == ActivationFunction.Tanh:
        A = tanh(Z)
    else:
        raise ValueError("Nieznana funkcja aktywacji")

    return A, (A_prev,Z)



def forward_propagation(X, parameters, function_activation_order):
    activations_history = []
    A = X

    # Iteracja przez warstwy sieci
    for layer in range(len(parameters[0])):
        A_prev = A 
        weights = parameters[0][layer]
        bias = parameters[1][layer]
        activation = function_activation_order[layer]
        A, activation_history = activation_function_forward(A_prev, weights, bias, activation)
        activations_history.append(activation_history)
    return A, activations_history


def linear_backward(dZ, A_prev, weights):
    m = A_prev.shape[1]
    dW = (1 / m) * (dZ @ A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = weights.T @ dZ
    return dA_prev, dW, db



def calculate_layer_gradients(dA, activations_history, weights, activation):
    A_prev, Z = activations_history  # Rozpakowanie krotki
    if activation == ActivationFunction.Relu:
        dZ = relu_backward(dA, Z)
    elif activation == ActivationFunction.Softmax:
        dZ = softmax_backward(dA, Z)
    elif activation == ActivationFunction.Sigmoid:
        dZ = sigmoid_backward(dA, Z)
    elif activation == ActivationFunction.Tanh:
        dZ = tanh_backward(dA, Z)
    dA_prev, dW, db = linear_backward(dZ, A_prev, weights)
    return dA_prev, dW, db




def backward_propagation(Y, actual_layers, activations_history, parameters, function_activation_order):
    gradients = {}
    L = len(activations_history)  # Liczba warstw
    dA = - ((Y / (actual_layers + 1e-4)) - ((1 - Y) / (1 - actual_layers + 1e-4)))

    for l in reversed(range(L)):
        weights = parameters[0][l]
        current_Z = activations_history[l]
        current_activation = function_activation_order[l]
        #print(current_Z[0].shape)
        #print(current_Z[1].shape)

        #print(current_activation)
        dA_prev, dW, db = calculate_layer_gradients(dA, current_Z, weights, current_activation)
        gradients[f"dA{l}"] = dA_prev
        gradients[f"dW{l}"] = dW
        gradients[f"db{l}"] = db
        dA = dA_prev  # Aktualizacja dA dla poprzedniej warstwy

    return gradients





def compute_cost(model_results, Y):
    number_of_data = Y.shape[1]
    cost = (-1 / number_of_data) * np.sum(Y * np.log(model_results + 1e-4))
    return cost


def get_function_activation_order(layer_dims, want_defult_setup=True, input = None):
    if want_defult_setup == True:
        function_activation_order =  [ActivationFunction.Relu for x in range(len(layer_dims)-2)]
        function_activation_order.append(ActivationFunction.Softmax)
        return function_activation_order 
    else:
        translate_function_order(layer_dims,input)


def translate_function_order(layer_dims,function_activation_order):
    activation_order = []
    for tupl in function_activation_order:
        if type(tupl[0]) is not ActivationFunction or type(tupl[1]) is not int:
            raise Exception("Niepoprawne dane wejściowe. Dane powinny być w postaci List(tuple(ActivationFunction,int))")
        for _ in range(tupl[1]):
            activation_order.append(tupl[0])
    if len(activation_order) != len(layer_dims)-2:
        raise Exception("Niepoprawne długosc tablicy wejsciowej. Funkcji aktywacji powinny byc tyle ile warstw ukrytych, gdyz ostatnia musi byc softmax")
    activation_order.append(ActivationFunction.Softmax) 
    return activation_order

def update_parameters(parameters, grads, learning_rate):
   
    weights = parameters[0]
    bias = parameters[1]
    L = len(weights)
    for l in range(L):
        weights[l] -= learning_rate * grads[f"dW{l}"]
        bias[l] -= learning_rate * grads[f"db{l}"]
    return weights,bias






def neural_network(X, Y, layers_dims, function_activation_order , learning_rate, epoka):
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
    parameters = initialize_parameters(layers_dims,InitializationMethod.RANDOM)
    print(str(type(parameters)))
    print(parameters[0][0].shape)
    print(parameters[1][0].shape)
    # Loop (gradient descent)
    for i in range(0, epoka):

        # Forward propagation
        actual_layers,  activations_history = forward_propagation(X, parameters, function_activation_order )

    
        #print(np.shape(Y))


        #print(actual_layers)
        # Compute cost
        cost = compute_cost(actual_layers, Y)
        


        # Backward propagation


        grads = backward_propagation(Y,actual_layers,  activations_history, parameters, function_activation_order)
        
       
        

        # Update parameters
        parameters = update_parameters(parameters, grads, learning_rate)
        
        # Print the cost every 100 training example
        if i % 10 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
            costs.append(cost)
        
            
    return parameters


def save_array_as_csv(array, file_path):
    # Jeśli tablica jest jednowymiarowa, przekształć ją w tablicę 2D (jeden wiersz)
    if array.ndim == 1:
        array = array.reshape(1, -1)

    try:
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(array)
        print(f"Plik CSV został pomyslnie zapisany jako {file_path}")
    except Exception as e:
        print(f"Błąd podczas zapisywania pliku CSV: {e}")

def check_test(X, params, order):
    actual_layers,  activations_history = forward_propagation(X, params, order)
    return actual_layers,  activations_history


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


np.set_printoptions(threshold=sys.maxsize)

layers_dims = [784, 392, 196, 98, 49, 10] 

#print(translate_function_order(layers_dims, [(ActivationFunction.Relu,2),(ActivationFunction.Tanh,2)]))
order = get_function_activation_order(layers_dims)




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
test_label = list_of_datas[3]
save_array_as_csv(test_label, "Test.csv")
test_data = np.transpose(list_of_datas[2])
test_label = np.transpose(list_of_datas[3])

parameters = neural_network(train_data, train_label, layers_dims,order, learning_rate=0.0002, epoka=10)



predictions, _ = check_test(test_data, parameters,order)

print("Macierz odpowiedzi ma rozmiary: " + str(np.shape(predictions)))
print(str(np.max(predictions)))
predictions = translate_matrix_of_probabilities_to_matrix_of_answers(np.transpose(predictions))
print(str(matrix_comparison(predictions,np.transpose(test_label))))

