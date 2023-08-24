import numpy as np
import pandas as pd
import sklearn.model_selection
from sklearn import linear_model
from sklearn.utils import shuffle

data = pd.read_csv('./student-mat.csv', sep=';')

data_cleaned = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"
# we got x
X = np.array(data_cleaned.drop([predict], axis=1))
y = np.array(data_cleaned[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)

print(acc)

print("Co: \n", linear.coef_)
print("Intercept: \n", linear.intercept_)

predictions = linear.predict(x_test)


for x in range(len(predictions)):
    print(predictions[x], x_test[x])