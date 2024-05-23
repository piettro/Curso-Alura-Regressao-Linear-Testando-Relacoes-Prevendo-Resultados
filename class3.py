import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

data = pd.read_csv('Data/Consumo_cerveja.csv',sep=';')

ax = sns.pairplot(data, y_vars='consumo', x_vars=['temp_min', 'temp_media', 'temp_max', 'chuva', 'fds'], kind='reg')
ax.fig.suptitle('Variables Dispersion', fontsize=20, y=1.05)
plt.show()

ax = sns.jointplot(x="temp_max", y="consumo", data=data, kind='reg')
ax.fig.suptitle('Dispersion - Consumption X Temperature', fontsize=18, y=1.05)
ax.set_axis_labels("Max Temperature", "Beer Consumption", fontsize=14)
plt.show()

ax= sns.lmplot(x="temp_max", y="consumo", data=data, hue='fds', markers=['o','*'], legend=False)
ax.fig.suptitle('Linear Regression - Consumption X Temperature', fontsize=16, y=1.02)
ax.set_xlabels("Max Temperature (°C)", fontsize=14)
ax.set_ylabels("Beer Consumption (litros)", fontsize=14)
ax.add_legend(title='Weekend')
plt.show()

ax= sns.lmplot(x="temp_max", y="consumo", data=data, col='fds')
ax.fig.suptitle('Linear Regression - Consumption X Temperature', fontsize=16, y=1.02)
ax.set_xlabels("Max Temperature (°C)", fontsize=14)
ax.set_ylabels("Beer Consumption (litros)", fontsize=14)
ax.add_legend(title='Weekend')
plt.show()
