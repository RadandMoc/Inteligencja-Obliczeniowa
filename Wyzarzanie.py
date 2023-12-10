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



readData = pd.read_csv("Dane_TSP_127.csv",sep=";")
readData = ChangeCommaToPoint(readData)
print(readData.iloc[1,1])