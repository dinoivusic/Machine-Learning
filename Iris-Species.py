
"""
Basic Logistic Regression program used to classify Iris speices
"""
#Import the dependencies
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#Dataset load
data = sns.load_dataset("iris")
data.head()

#prepare training set

# X will be feature values, all cols except last one
X=data.iloc[:,:-1]

#y will be target values, only last column of df

y = data.iloc[:, -1]

# Plot relation of each feature with each species

plt.xlabel('Features')
plt.ylabel('Species')

pltX = data.loc[:, 'sepal_length']
pltY = data.loc[:, 'species']
plt.scatter(pltX, pltY, color='red', label='sepal_length')

pltX = data.loc[:, 'sepal_width']
pltY = data.loc[:, 'species']
plt.scatter(pltX, pltY, color='blue', label='sepal_width')

pltX = data.loc[:, 'petal_length']
pltY = data.loc[:, 'species']
plt.scatter(pltX, pltY, color='green', label='petal_length')

pltX = data.loc[:, 'petal_length']
pltY = data.loc[:, 'species']
plt.scatter(pltX, pltY, color='black', label='petal_length')

plt.legend(loc=4, prop={'size':9})
plt.show()

#Split the data (80% tarining and 2-% test)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

model= LogisticRegression()
model.fit(X_train, y_train)

#Testing the model
predictions = model.predict(X_test)

#Checking the accuracy, recall, f1-score
print(classification_report(y_test,predictions))
print(accuracy_score(y_test, predictions))





