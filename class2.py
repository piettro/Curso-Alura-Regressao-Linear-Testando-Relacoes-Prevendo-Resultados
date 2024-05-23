import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

data = pd.read_csv('Data/Consumo_cerveja.csv',sep=';')
print(data.head(10))

##Corr table
fig, ax = plt.subplots(figsize=(20,6))
ax = data['consumo'].plot()
plt.show()

ax = sns.boxplot(data=data['consumo'], orient='v', width=0.2)
ax.figure.set_size_inches(12,6)
ax.set_title('Beer Consumption', fontsize=20)
ax.set_ylabel('Liters', fontsize=16)
plt.show()

ax = sns.boxplot(y='consumo', x='fds', data=data, orient='v', width=0.2)
ax.figure.set_size_inches(12,6)
ax.set_title('Beer Consumption', fontsize=20)
ax.set_ylabel('Liters', fontsize=16)
ax.set_xlabel('Weekend', fontsize=16)
plt.show()

ax = sns.histplot(data['consumo'], kde=True)
ax.figure.set_size_inches(12, 6)
ax.set_title('Frequency Distribution', fontsize=20)
ax.set_ylabel('Beer Consumption (Liters)', fontsize=16)
plt.show()