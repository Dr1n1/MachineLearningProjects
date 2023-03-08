import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style


# Here we told the machines to take the data from the student-mat file.
data = pd.read_csv("student-mat.csv", sep=";")

# print(data.head())
# where as here we told it to get specific data/columns from the file.
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

# print(data.head())


predict = "G3"


X = np.array(data.drop([predict], 1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)


best = 0
for _ in range(20): # trains 20 models
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)

    acc = linear.score(x_test, y_test)
    print(acc) # Here we print the predicted accuracy.

    if acc > best: # here we are saving a model if this model performs better than the last one.
        best = acc
        with open("studentmodel.pickle", "wb") as f: # Here we saved our model
            pickle.dump(linear, f)
        print("The best model so far: ", best)

pickle_in = open("studentmodel.pickle", "rb") # Here we are loading our model (not retraining)
linear = pickle.load(pickle_in)

print("Coef: \n", linear.coef_) # Here are printed all the coefficents of m, in 5 dimentional space.
print('Intercept: \n', linear.intercept_) # starting line of Y.

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print (predictions[x], x_test[x], y_test[x])

#this part gives a visual grid view of the data
p = 'studytime'
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()