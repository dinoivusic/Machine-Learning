import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import f1_score,precision_score,recall_score,classification_report,confusion_matrix, accuracy_score
from sklearn import svm
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import pickle
df = pd.read_csv('https://raw.githubusercontent.com/mrdbourke/zero-to-mastery-ml/master/data/heart-disease.csv')
df.head()

X = df.drop('target', 1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X.shape,y.shape
clf = RandomForestClassifier()
clf.fit(X_train,y_train)
y_preds = clf.predict(X_test)
np.random.seed(42)
print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))


model_dict = {"RandomForestClassifier": RandomForestClassifier(),
             "SVC": svm.SVC(), "LogisticRegression": LogisticRegression(), "LinearSVC":svm.LinearSVC()}
results={}

for model_name, model in model_dict.items():
    model.fit(X_train, y_train)
    results[model_name] = model.score(X_test, y_test)

results

results_df = pd.DataFrame(results.values(), results.keys(), columns=['Accuracy'])
log_reg_grid = {"C": np.logspace(-4, 4, 20),
                "solver": ["liblinear"]}
results_df
rs_log_grid = RandomizedSearchCV(estimator=LogisticRegression(), param_distributions=log_reg_grid, cv=5, n_iter=5, verbose=2)
rs_log_grid.fit(X_test, y_preds)
rs_log_grid.score(X_test, y_preds)
clf = LogisticRegression()
import seaborn as sns
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_preds = clf.predict(X_test)
confusion = confusion_matrix(y_test, y_preds)
confusion
ax = sns.heatmap(confusion,annot=True)
classification_report(y_test, y_preds)

#report = {'precision':precision_score(), 'recall': recall_score, 'f1_score': f1_score, 'accuracy':accuracy_score}
precision = precision_score(y_test, y_preds)

recall = recall_score(y_test, y_preds)
f1 = f1_score(y_test, y_preds)
accuracy = accuracy_score(y_test, y_preds)
#print(precision, recall, f1, accuracy)
classification = classification_report(y_test, y_preds)
#print(classification)
pd.DataFrame(classification_report(y_test, y_preds, output_dict=True))
cross_val_acc = np.mean(cross_val_score(clf, X, y, scoring='precision', cv=5))
cross_val_acc
cross = cross_validate(clf, X, y, scoring=['accuracy','precision', 'recall'], cv=5)
cross
model_saved = pickle.dump(clf,open('saved_model.pkl', 'wb'))
loaded = pickle.load(open('saved_model.pkl', 'rb'))
