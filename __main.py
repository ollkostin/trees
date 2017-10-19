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
learnX = data_array[1:600, 1:21]
learnY = data_array[1:600, 0]

# print(learnX)
# print(learnY)

# fit a CART model to the data
model = DecisionTreeClassifier()
model.fit(learnX, learnY)
# make predictions
learnExpected = learnY
learnPredicted = model.predict(learnX)
# summarize the fit of the model
print(metrics.classification_report(learnExpected, learnPredicted))
print(metrics.confusion_matrix(learnExpected, learnPredicted))

tree.export_graphviz(model, 'learnModel.dot')
check_call(['dot', '-Tpng', 'learnModel.dot', '-o', 'learnModel.png'])

# testing model
testX = data_array[601:, 1:21]
testY = data_array[601:, 0]

# fit a CART model to the data
model.fit(testX, testY)

testExpected = testY
testPredicted = model.predict(testX)
print(metrics.classification_report(testExpected, testPredicted))
print(metrics.confusion_matrix(testExpected, testPredicted))

tree.export_graphviz(model, 'controlModel.dot')
check_call(['dot', '-Tpng', 'controlModel.dot', '-o', 'controlModel.png'])
