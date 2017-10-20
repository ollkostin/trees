import csv
from sklearn import metrics, tree, model_selection
from subprocess import check_call
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import numpy as np

# read data from file
with open('p/kredit_3.csv', 'r') as dest_f:
    data_iter = csv.reader(dest_f)
    data = [data for data in data_iter]
data_array = np.asarray(data)
X = data_array[1:, 1:21]
Y = data_array[1:, 0]

learnX, testX, learnY, testY = model_selection.train_test_split(X, Y, train_size=0.20, random_state=100)

# fit a CART model to the data
model = DecisionTreeClassifier(max_leaf_nodes=6,  max_depth=3, min_samples_split=2, splitter="best",
                               min_weight_fraction_leaf=0.)
model.fit(learnX, learnY)
# make predictions
expected = testY
predicted = model.predict(testX)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
print(metrics.accuracy_score(expected, predicted) * 100)

tree.export_graphviz(model, 'tree.dot')
check_call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png'])
