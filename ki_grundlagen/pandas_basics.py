import pandas as pd

data = {
    'Name': ['Anna', 'Max', 'Lisa'],
    'Alter': [25, 30, 22],
    'Stadt': ['Berlin', 'München', 'Hamburg']
}

df = pd.DataFrame(data)
print(df)