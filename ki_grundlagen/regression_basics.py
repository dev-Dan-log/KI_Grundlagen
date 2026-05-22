import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt


data = {
    'Größe': [50, 80, 120, 45, 100, 150, 60, 90],
    'Zimmer': [2, 3, 4, 1, 3, 5, 2, 3],
    'Preis€': [150000, 220000, 350000, 120000, 280000, 450000, 170000, 250000]
}

df = pd.DataFrame(data)

X = df[['Größe', 'Zimmer']]
y = df['Preis€']
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
model = LinearRegression()
model.fit(X, y)


slope_groesse = model.coef_[0]
slope_zimmer = model.coef_[1]
intercept = model.intercept_
print(f"Gleichung: Preis = {slope_groesse:.2f} * Größe + {slope_zimmer:.2f} * Zimmer + {intercept:.2f}")


X_new = np.array([[65, 2], [85, 3]])
predictions = model.predict(X_new)
print(f"Vorhersage für die neuen Wohnungen: {predictions}")


plt.scatter(df['Größe'], y, color='blue', label='Echte Datenpunkte')

plt.scatter(df['Größe'], model.predict(X), color='red', marker='x', label='Modell-Vorhersagen')
plt.xlabel('Größe in qm')
plt.ylabel('Preis in €')
plt.legend()
plt.show()
print(X_scaled)