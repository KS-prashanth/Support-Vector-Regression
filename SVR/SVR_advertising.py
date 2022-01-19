import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings('ignore')


dataset = pd.read_csv('advertising.csv')
# Removing the unnecessary column if there is a index column
# dataset.drop(['Unnamed: 0'], axis = 1, inplace = True)
print(dataset.head())
X = X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1:].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
X_sc = StandardScaler()
y_sc = StandardScaler()
X_train = X_sc.fit_transform(X_train)
y_train = y_sc.fit_transform(y_train)

from sklearn.svm import SVR
regrassor = SVR(kernel = 'rbf')
regrassor.fit(X_train, y_train)

y_pred = regrassor.predict(X_sc.transform(X_test))
y_pred = y_sc.inverse_transform(y_pred)

y_test = y_test.flatten()

df = pd.DataFrame({'Predicted value': y_pred, 'Real Value': y_test})
print(df)

from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test, y_pred))

# new_pred=regrassor.predict(X_sc.transform([[11.2,0.28,0.56,1.9,0.075,17,60,0.998,3.16,0.58,9.8]]))
# print(y_sc.inverse_transform(new_pred))
