import csv
from subprocess import check_call

import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
# read data from file
from sklearn.tree import export_graphviz

with open('p/kredit_3.csv', 'r') as dest_f:
    data_iter = csv.reader(dest_f)
    data = [data for data in data_iter]

data_array = np.asarray(data)
learnX = data_array[1:500, 1:21]
learnY = data_array[1:500, 0]
testX = data_array[502:, 1:21]
testY = data_array[502:, 0]

# fit model to the data
model = RandomForestClassifier(5)
model.fit(learnX, learnY)
# make predictions
expected = testY
predicted = model.predict(testX)
# summarize the fit of the model

print(metrics.accuracy_score(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

i = 1
for m in model.estimators_:
    prefix = 'forest_tree' + str(i)
    export_graphviz(m, prefix + '.dot')
    check_call(['dot', '-Tpng', prefix + '.dot', '-o', prefix + '.png'])
    i += 1
