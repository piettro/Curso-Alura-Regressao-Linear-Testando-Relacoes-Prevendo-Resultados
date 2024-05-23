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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2811)

model = LinearRegression()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
y_predict_train = model.predict(X_train)
residual = y_train - y_predict_train

print(f'R² = {model.score(X_train, y_train)}')
print(f'R² = {metrics.r2_score(y_test, y_predict)}')

def return_model_predict(input_model):
    predict_with_input_model = model.predict(input_model)[0]
    print(f'The predict with input: {input_model} is {predict_with_input_model}')

temp_max = 30.5
rain = 12.2
weekend = 0
input_model = [[temp_max, rain, weekend]]
return_model_predict(input_model)

index = ['Intercept','Temp Max','Rain (mm)', 'Weekend']
df = pd.DataFrame(data=np.append(model.intercept_, model.coef_), index=index, columns=['Parameters'])
print(df.head())

ax = sns.scatterplot(x=y_predict_train, y=y_train)
ax.figure.set_size_inches(12, 6)
ax.set_title('Predict X Real', fontsize=18)
ax.set_xlabel('Beer Consumption (Liters) - Predict', fontsize=14)
ax.set_ylabel('Beer Consumption (Liters) - Real', fontsize=14)
plt.show()

ax = sns.scatterplot(x=y_predict_train, y=residual, s=150)
ax.figure.set_size_inches(20, 8)
ax.set_title('Residuals X Predict', fontsize=18)
ax.set_xlabel('Beer Consumption (Liters) - Predict', fontsize=14)
ax.set_ylabel('Resíduos', fontsize=14)
plt.show()

ax = sns.scatterplot(x=y_predict_train, y=residual**2, s=150)
ax.figure.set_size_inches(20, 8)
ax.set_title('Residuals X Predict', fontsize=18)
ax.set_xlabel('Beer Consumption (Liters) - Predict', fontsize=14)
ax.set_ylabel('Residuals^2', fontsize=14)
plt.show()

ax = sns.histplot(residual)
ax.figure.set_size_inches(12, 6)
ax.set_title('Distribuição de Frequências dos Resíduos', fontsize=18)
ax.set_xlabel('Liters', fontsize=14)
plt.show()