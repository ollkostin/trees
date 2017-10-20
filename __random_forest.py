import csv
from subprocess import check_call

import numpy as np
from sklearn import metrics, model_selection
from sklearn.ensemble import RandomForestClassifier
# read data from file
from sklearn.tree import export_graphviz

with open('p/kredit_3.csv', 'r') as dest_f:
    data_iter = csv.reader(dest_f)
    data = [data for data in data_iter]

data_array = np.asarray(data)
learnX = data_array[1:500, 1:21]
learnY = data_array[1:500, 0]
X = data_array[1:, 1:21]
Y = data_array[1:, 0]

learnX, testX, learnY, testY = model_selection.train_test_split(X, Y, train_size=0.20, random_state=100)

# fit model to the data
model = RandomForestClassifier(n_estimators=330, oob_score=True)
model.fit(learnX, learnY)
# make predictions
expected = testY
predicted = model.predict(testX)
# summarize the fit of the model

print(metrics.accuracy_score(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
print(model.oob_score_ * 100)

# i = 1
# for i in range(1, 300):
#     models = RandomForestClassifier(n_estimators=i, oob_score=True)
#     expected = testY
#     predicted = model.predict(testX)
#     models.fit(learnX, learnY)
#     print(str(i) + " " + str(model.oob_score_ * 100))

# i = 1
# for m in model.estimators_:
#     prefix = 'forest_tree' + str(i)
#     export_graphviz(m, prefix + '.dot')
#     check_call(['dot', '-Tpng', prefix + '.dot', '-o', prefix + '.png'])
#     i += 1
