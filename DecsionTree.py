import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier

# load data by using iris data
iris = datasets.load_iris()
X = iris.data
y = iris.target

# train model
clf = DecisionTreeClassifier(criterion='entropy',max_depth=4)
#fitting model
clf.fit(X, y)


# draw

print(X)

from IPython.display import Image  
from sklearn import tree
import pydotplus 

dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=iris.feature_names,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = pydotplus.graph_from_dot_data(dot_data)  
Image(graph.create_png()) 

#calculate the entropy

import math

class_a = iris.target
fea = iris.data[0]
count = len(fea)

dic = {}
for r in fea:
    if not r in dic:
        dic[str(r)] = {'Iris-setosa':0,'Iris-versicolor':0, 'Iris-virginica':0}

for i in range(count):
    t_d = dic[str(fea[i])]
    t_d[class_a[i]] += 1
    dic[str(fea[i])] = t_d

print(len(dic))

ent = 0.0
for r in dic:
    p0 = dic[r]['Iris-setosa']
    p1 = dic[r]['Iris-versicolor']
    p2 = dic[r]['Iris-virginica']
    p_count = p0+p1+p2
    p0 = p0/p_count
    p1 = p1/p_count
    p2 = p2/p_count
    c1 = 0.0
    c2 = 0.0
    c3 = 0.0
    if not p0 == 0:
        p0 = p0/p_count
        c0 = p0*math.log(p0)/math.log(2)
    if not p1 == 0:
        p1 = p1/p_count
        c1 = p1*math.log(p1)/math.log(2)
    if not p2 == 0:
        p2 = p2/p_count
        c2 = p2*math.log(p2)/math.log(2)
    temp_ent = -(c1+c2+c3)
    ent += (p_count/count)*temp_ent/math.log(2)

print(ent)
