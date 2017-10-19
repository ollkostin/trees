import csv
from sklearn import metrics, tree
from subprocess import check_call
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import numpy as np

# read data from file
with open('p/kredit_3.csv', 'r') as dest_f:
    data_iter = csv.reader(dest_f)
    data = [data for data in data_iter]
data_array = np.asarray(data)
learnX = data_array[1:500, 1:21]
learnY = data_array[1:500, 0]
testX = data_array[502:, 1:21]
testY = data_array[502:, 0]

# fit a CART model to the data
model = DecisionTreeClassifier()
model.fit(learnX, learnY)
# make predictions
expected = testY
predicted = model.predict(testX)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

tree.export_graphviz(model, 'tree.dot')
check_call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png'])
