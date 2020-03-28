import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('car_integer_exceptY.csv')

df.head()

X_train = df.loc[:, 'buying': 'safety']
Y_train = df.loc[:, 'values']

tree = DecisionTreeClassifier(max_leaf_nodes=3, random_state= 0)
tree.fit(X_train, Y_train)
prediction = tree.predict([[4,2,4,4,2,2]])

print(prediction)


