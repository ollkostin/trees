import csv
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
import numpy as np

with open('/p/kredit_3.csv', 'r') as dest_f:
    data_iter = csv.reader(dest_f,
                           delimiter=',',
                           quotechar='"')
    data = [data for data in data_iter]
data_array = np.asarray(data)
x = data_array[1:, :]
y = data_array[1:, 0]
# print(x)
# print(y)


# fit a CART model to the data
model = DecisionTreeClassifier()
model.fit(x, y)
print(model)
# make predictions
expected = y
predicted = model.predict(x)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
# tree.export_graphviz(model, out_file='tree.dot')
