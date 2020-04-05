import pandas as pd
import numpy as np
import quandl, math, datetime, pickle
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

df = quandl.get('WIKI/GOOGL')
print(df.head())
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume',]]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close']*100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open']*100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
print(df.tail())
forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.01*len(df))) # predicting 10% of data frame
df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'],1))
#Splitting X into 2 frames where X is up to - forecast_out and lately is
# last - forecast out rows
X = preprocessing.scale(X)
X_lately = X[-forecast_out:] # in this case len is 35 we are predicting against and we dont have y value for that+
X = X[:-forecast_out] # in this case len is 3354

df.dropna(inplace=True)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(len(y_train))
print(len(X_train))
clf = LinearRegression(n_jobs=10)
clf.fit(X_train,y_train)

### Saving the classifier
# with open('linearRegression.pkl', 'wb') as f:
#     pickle.dump(clf,f) # dumping the classifier into f
#
# pickle_in = open('linearRegression.pkl', 'rb')
# clf = pickle.load(pickle_in)

accuracy = clf.score(X_test,y_test)
#print(accuracy)

forecast_set = clf.predict(X_lately)

#print(forecast_set, accuracy,forecast_out)

df['Forecast'] = np.nan

last_day = df.iloc[-1].name
print(type(last_day))
last_unix = last_day.timestamp()
oneday = 86400
next_unix = last_unix + oneday

#iterating through predictions for last days and setting those as vlaues in df, making future features not a number
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += oneday
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

print(df.tail())

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
