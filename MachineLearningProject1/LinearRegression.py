import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

# Here we told the machines to take the data from the student-mat file.
data = pd.read_csv("student-mat.csv", sep=";")

# print(data.head())
# where as here we told it to get specific data/columns from the file.
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

# print(data.head())


predict = "G3"


X = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)

acc = linear.score(x_test, y_test)
print(acc) # Here we print the predicted accuracy.
print("Coef: \n", linear.coef_) # Here are printed all the coefficents of m, in 5 dimentional space.
print('Intercept: \n', linear.intercept_) # starting line of Y.

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print (predictions[x], x_test[x], y_test[x])