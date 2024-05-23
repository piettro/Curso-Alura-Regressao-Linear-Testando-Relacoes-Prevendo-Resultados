import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

data = pd.read_csv('Data/Consumo_cerveja.csv',sep=';')

y = data['consumo']
X = data[['temp_max', 'chuva', 'fds']]
X_2 = data[['temp_media', 'chuva', 'fds']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2811)
X2_train, X2_test, y2_train, y2_test = train_test_split(X_2, y, test_size=0.3, random_state=2811)

model_1 = LinearRegression()
model_1.fit(X_train, y_train)
y_predict = model_1.predict(X_test)
y_predict_train = model_1.predict(X_train)
residual = y_train - y_predict_train

print(f'=========== Model 1 ===========')
print(f'R² = {model_1.score(X_train, y_train)}')
print(f'R² = {metrics.r2_score(y_test, y_predict)}')
print(f'EQM {metrics.mean_squared_error(y_test, y_predict)}')
print(f'REQM {np.sqrt(metrics.mean_squared_error(y_test, y_predict))}')

model_2 = LinearRegression()
model_2.fit(X2_train, y2_train)
y2_predict = model_2.predict(X2_test)
y2_predict_train = model_2.predict(X2_train)
residual_2 = y2_train - y2_predict_train

print(f'=========== Model 2 ===========')
print(f'R² = {model_2.score(X2_train, y2_train)}')
print(f'R² = {metrics.r2_score(y2_test, y2_predict)}')
print(f'EQM {metrics.mean_squared_error(y2_test, y2_predict)}')
print(f'REQM {np.sqrt(metrics.mean_squared_error(y2_test, y2_predict))}')