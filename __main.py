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

print(learnX)
print(learnY)

# fit a CART model to the data
learnModel = DecisionTreeClassifier()
learnModel.fit(learnX, learnY)
# make predictions
learnExpected = learnY
learnPredicted = learnModel.predict(learnX)
# summarize the fit of the model
print(metrics.classification_report(learnExpected, learnPredicted))
print(metrics.confusion_matrix(learnExpected, learnPredicted))

# testing model
testX = data_array[601:, 1:21]
testY = data_array[601:, 0]

# fit a CART model to the data
learnModel.fit(testX, testY)

testExpected = testY
testPredicted = learnModel.predict(testX)
print(metrics.classification_report(testExpected, testPredicted))
print(metrics.confusion_matrix(testExpected, testPredicted))

tree.export_graphviz(learnModel, 'learnModel.dot')
check_call(['dot', '-Tpng', 'learnModel.dot', '-o', 'learnModel.png'])
