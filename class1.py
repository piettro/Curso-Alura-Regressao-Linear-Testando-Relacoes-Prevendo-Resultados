import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('Data/Consumo_cerveja.csv',sep=';')
print(data.head(10))

##Corr table
print(data.drop('data',axis = 1).corr().round(4))