#!/usr/bin/env python
# coding: utf-8

# # Predicting heart disease using Machine Learning
# 
# Predicting if patient has heart disease or not based on the data and medical attributes
# 
# Steps and approach:
# 1. Problem definition
# 2. Data
# 3. Evaluation
# 4. Features 
# 5. Modelling
# 6. Experimentation
# 
# ## 1. Problem defintion 
# 
# In a statement > 
# Given clinical parameters can we predict if patient has heart disease or not
# 
# ## 2. Data
# 
# Contains 14 attributes and original data came from UCI ML Repository
# 
# ## 3. Evaluation
# 
# > If we can reach 95% accuray and predict if patient has heart disease throguh proof of concept and if so we will pursue the project
# 
# ## 4. Features
# 
# This is where you get dfferent info about each feature in your data
# 
# **Data dictionary** 
# 
# * age - age in years
# * sex - (1 = male; 0 = female)
# * cp - chest pain type
# * 0: Typical angina: chest pain related decrease blood supply to the heart
# * 1: Atypical angina: chest pain not related to heart
# * 2: Non-anginal pain: typically esophageal spasms (non heart related)
# * 3: Asymptomatic: chest pain not showing signs of disease
# * trestbps - resting blood pressure (in mm Hg on admission to the hospital) anything above 130-140 is typically cause for concern
# * chol - serum cholestoral in mg/dl
# * serum = LDL + HDL + .2 * triglycerides
# * above 200 is cause for concern
# * fbs - (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
# * '>126' mg/dL signals diabetes
# * restecg - resting electrocardiographic results
# * 0: Nothing to note
# * 1: ST-T Wave abnormality can range from mild symptoms to severe problems signals non-normal heart beat
# * 2: Possible or definite left ventricular hypertrophy Enlarged heart's main pumping chamber
# * thalach - maximum heart rate achieved
# * exang - exercise induced angina (1 = yes; 0 = no)
# * oldpeak - ST depression induced by exercise relative to rest looks at stress of heart during excercise unhealthy heart will stress more
# * slope - the slope of the peak exercise ST segment
# * 0: Upsloping: better heart rate with excercise (uncommon)
# * 1: Flatsloping: minimal change (typical healthy heart)
# * 2: Downslopins: signs of unhealthy heart
# * ca - number of major vessels (0-3) colored by flourosopy colored vessel means the doctor can see the blood passing throughthe more blood movement the better (no clots)
# * thal - thalium stress result
# * 1,3: normal
# * 6: fixed defect: used to be defect but ok now
# * 7: reversable defect: no proper blood movement when excercising
# * target - have disease or not (1=yes, 0=no) (= the predicted attribute)

# Regular EDA (exploratory data analysis) and plotting libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Models from SciKit
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Model evaluations 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score,recall_score,f1_score

# Load Data
df = pd.read_csv('./heart-disease.csv')
df.shape


# # Data exploration (EDA)
# 
# The goal here is to find out more about data and become subject matter expert on the dataset that you are working on
# 
# 1. What questions are we solving?
# 2. What kind of data do we have? And how we treat different types?
# 3. What's missing from the data and how to deal with it?
# 4. Where are  the outliers and why to care about them?
# 5. How can you add change or remove features to get mroe out of data?

# How many of each class is there? 1- has HD, 0 - doesnt have
df['target'].value_counts()
df['target'].value_counts().plot(kind='bar', color=['green', 'blue'])
df.isna().sum(), df.info()
df.sex.value_counts()

# Compare target column with sex colm with crosstab
pd.crosstab(df.target, df.sex)

#create a plot of crosstab
pd.crosstab(df.target, df.sex).plot(kind = 'bar', figsize=(10,6), color=['salmon','lightblue'])
plt.title('Heart disease per sex')
plt.xlabel(' 0 = no disease, 1 = disease')
plt.ylabel('Amount')
plt.legend(['Female','Male'])

pd.crosstab(df.target,df.cp).plot(kind='bar')

# Creating another figure
plt.figure(figsize=(10,6))
# Scatter with positive examples
plt.scatter(df.age[df.target==1], df.thalach[df.target==1],c='lightblue')
#scatter with negatives
plt.scatter(df.age[df.target==0], df.thalach[df.target==0],c='salmon')
#Add some helpful info
plt.title('Heart disease in Age and Max Heart rate')
plt.xlabel('Age')
plt.ylabel('Max heart rate')
plt.legend(['Disease', 'No disease'])

#Check the distribution of age with histogram
df.age.hist();

# ## Heart Disease Frequency per Chest Pain type
# 0: Typical angina: chest pain related decrease blood supply to the heart
# 1: Atypical angina: chest pain not related to heart
# 2: Non-anginal pain: typically esophageal spasms (non heart related)
# 3: Asymptomatic: chest pain not showing signs of disease

pd.crosstab(df.cp,df.target)
# Make crosstab more visual
pd.crosstab(df.cp, df.target).plot(kind='bar',figsize=(10,6),color=['salmon','lightblue'])
plt.title('Heart Disesae Frequency per chest pain type')
plt.xlabel('Chest Pain type')
plt.ylabel('Amount')
plt.legend(['No disease', 'Disease']);

# Make a correlation matrix
df.corr()
# Lets set it up on a heatmap
corr_matrix = df.corr()
fig, ax = plt.subplots(figsize=(15,10))
ax = sns.heatmap(corr_matrix, annot=True, linewidths=0.5,fmt ='.2f', cmap='YlGnBu')
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)

# 5. Modelling

# Splitting data
X = df.drop('target', 1)
y = df['target']
# Split data into train adn test
np.random.seed(42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# # Picking a proper model and train it
# # hen test it on test data and use the patterns that were found in training
# #Trying 3 different ML models
# # 1. Logistic Regression
# # 2. K-Nearest Neighbours Classifier
# # 3. Random Forrest Classifier
# Put models in a dictionary

models = {'Logistic Regression': LogisticRegression(), 'KNN': KNeighborsClassifier(),
         'Random Forest': RandomForestClassifier()}
#Create a function to fit and score models

def fit_score(models, X_train, X_test, y_train, y_test):
    '''
    Fits and evaluates given machine learning models, using 3 different models
    X_train, X_test, - no labels for traing and testing data 
    y_train, y_test - labels for training and testing
    '''
    np.random.seed(42)
    #  Make a dict to store model socres
    model_score = {}
    # Loop through models
    for name, model in models.items():
        #Fit the model to data
        model.fit(X_train, y_train)
        #Evaluate model and append score to model score
        model_score[name] = model.score(X_test, y_test)
    return model_score


model_score = fit_score(models=models, X_train=X_train,X_test=X_test, y_train=y_train, y_test=y_test)

compare_models = pd.DataFrame(model_score, index=['accuracy'])
compare_models.T.plot.bar()


# 
# Check the following:
# 1. Hyperparameter tuning
# 2. Feature importance
# 3. Confusion Matrix
# 4. Cross validation
# 5.  Precision
# 6. Recall
# 7. F1 score
# 8. ROC curve
# 9. AUC - Area under curve

# Tune KNN 
train_scores = []
test_scores = []

# Create a list of different values for N-neighbours
neighbors = range(1,21)
#Setup KNN instance
knn = KNeighborsClassifier()
# Loop through different n_neighbours
for i in neighbors:
    knn.set_params(n_neighbors=i)
    #fit algorithm
    knn.fit(X_train, y_train)
    #update training score list
    train_scores.append(knn.score(X_train, y_train))
    test_scores.append(knn.score(X_test, y_test))
    
plt.plot(neighbors, train_scores, label='Train Scores')
plt.plot(neighbors, test_scores, label = 'Test Scores')
plt.xticks(np.arange(1,21))
plt.xlabel('Number of neighbors')
plt.ylabel('Model Score')
plt.legend()
print(f'Max test score on test data is:{max(test_scores)*100:.2f}%')


# ## Hyperparameter tuning with RandomizedSearchCV
# we are tuning:
# * Logistic Regression
# * Random Forest classifier
# using RandomizedSearchCV

log_reg_grid = {'C': np.logspace(-4, 4, 20), 'solver': ['liblinear']}
# Create hyperparameter grid for RandomForestClassifier
rf_grid = {'n_estimators': np.arange(10,1000,50), 'max_depth': [None,3,5,10],
          'min_samples_split': np.arange(2,20,2), 'min_samples_leaf': np.arange(1,20,2)}

# # Hyperparameter grids setup for each model is done, now use them to tune with RandomizedSearchCV
#Tune LogisticRegression
np.random.seed(42)
rs_log_reg = RandomizedSearchCV(LogisticRegression(), param_distributions=log_reg_grid, 
                                cv=5, n_iter=20, verbose=True)
# Fit random hyperparameter search model for Logistic Regression
rs_log_reg.fit(X_train, y_train)

rs_log_reg.score(X_test, y_test)

# # Tuning Random forest Classifier

np.random.seed(42)
rs_rf = RandomizedSearchCV(RandomForestClassifier(), param_distributions=rf_grid,
                          cv=5, n_iter=20, verbose=True)
#Fit random hyperparameter search model for RandomFporest classifier
rs_rf.fit(X_train, y_train)


# Evaluate the model
rs_rf.score(X_test, y_test)

# ## Hyperaparameters tuning using GridSearchcV
# Since LogisticRegression model provides the best scores so far, 
# we going to use GridSearchCV 

# Different hyperparameters for Logistic REgerssion 
log_reg_grid = {'C': np.logspace(-4, 4, 30), 'solver':['liblinear']}
gs_log_reg = GridSearchCV(LogisticRegression(), param_grid=log_reg_grid,
                         cv=5, verbose=True)
#Fit grid hyperparameter search model
gs_log_reg.fit(X_train, y_train)

# Evaluate grid searchLogistic regression model
gs_log_reg.score(X_test, y_test)

# ## Evaluating our tuned machine learning classifier, beyond accuracy
# 
# * ROC curve and AUC score
# * Confusion Matrix
# * Precision
# * Classification Report
# * Recall
# * F1-score
# 
# To make comparison and evaluate our trained model we need to make predictions

# Make predictions with tuned model
y_preds = gs_log_reg.predict(X_test)
print(confusion_matrix(y_test, y_preds))

sns.set(font_scale=1.5)
def conf_matrix_plot(y_test, y_preds):
    '''
    Plots a nice looking confusion matrix using seabron heatmap
    '''
    fig, ax = plt.subplots(figsize=(3,3))
    ax = sns.heatmap(confusion_matrix(y_test, y_preds),annot=True, cbar=False)
    plt.xlabel('True label')
    plt.ylabel('Predicted label')
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top -0.5)

conf_matrix_plot(y_test, y_preds)

print(classification_report(y_test, y_preds))

# # Calcualte evaluation metrics using cross validation
# Calculate precision, recall, f1score of our model using 'cross val score'
# 

# Best hyperparameters and new classifier 
clf = LogisticRegression(C=0.20433597178569418, solver='liblinear')

#Cross validate accuracy
cv_acc = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
cv_acc = np.mean(cv_acc)
#Cross validated precision
cv_precision = cross_val_score(clf, X, y, cv=5, scoring='precision')
cv_precision = np.mean(cv_precision)
#Cross validated recall
cv_recall = cross_val_score(clf, X, y, cv=5, scoring='recall')
cv_recall = np.mean(cv_recall)
#Cross validate f1-score
cv_f1 = cross_val_score(clf, X, y, cv=5, scoring='f1')
cv_f1 = np.mean(cv_f1)

print(cv_acc, cv_precision, cv_recall, cv_f1)

# Visualize our cross-validated metrics
cv_metrics = pd.DataFrame({'Accuracy': cv_acc, 'Precision':cv_precision,
                          'Recall': cv_recall, 'F1':cv_f1}, index=[0])
cv_metrics.T.plot.bar(title='Cross validated classification metrics', legend=False)

# ## Features Importance
# This is another way of asking in which features contributed most to the outcomes of the model and how did they contribute
# 
# Finding feature importance is different for each model
# * Finding feature importance for our Logistic Regression model

# Fit an instance of Logistic Regression
clf = LogisticRegression(C=0.20433597178569418, solver='liblinear')
clf.fit(X_train, y_train)

#Match coef of features to columns
feature_dict = dict(zip(df.columns, list(clf.coef_[0])))

# Visualization of feature importance
feature_df = pd.DataFrame(feature_dict, index=[0])
feature_df.T.plot.bar(title='Feature Importance', legend=False);




