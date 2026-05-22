import pandas as pd

data = {
    'Name': ['Anna', 'Max', 'Lisa', 'Tom', 'Sarah'],
    'Alter': [25, 30, 22, 35, 28],
    'Stadt': ['Berlin', 'München', 'Hamburg', 'Berlin', 'München']
}

df = pd.DataFrame(data)
d_a = df['Alter'].mean()
a_p = df['Alter'].max()
j_p = df['Alter'].min()
g_ue = df.describe()
print(d_a)
print(a_p)
print(j_p)
print(g_ue)