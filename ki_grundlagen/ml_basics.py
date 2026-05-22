import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = {
    'Alter' : [25, 35, 45, 20, 40, 30, 50, 22, 38, 42],
    'Gehalt' : [30000, 50000, 80000, 20000, 60000, 45000, 90000, 25000, 55000, 70000],
    'Kauft' : ['Nein', 'Ja', 'Ja', 'Nein', 'Ja', 'Nein', 'Ja', 'Nein', 'Ja', 'Ja']
}
df = pd.DataFrame(data)
X = df[['Alter', 'Gehalt']]
y = df['Kauft']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(f"Genauigkeit: {accuracy_score(y_test, predictions):.2f}")