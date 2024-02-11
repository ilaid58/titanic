import pandas as pd

df = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
#print(df.info())
#print(test.info())
#print(df.groupby('Cabin')['Cabin'].value_counts())
print(test.info())
