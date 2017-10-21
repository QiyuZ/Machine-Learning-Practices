from sklearn.ensemble import RandomForestClassifier as RFC
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

names = ['class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']
df = pd.read_csv('F:/wine.data', header=None, names=names)
target = []
a = np.array(df['class'])
for r in range(len(a)):
    if a[r] == 1:
        target.append(0)
    if a[r] == 2:
        target.append(1)
    if a[r] == 3:
        target.append(2)
target_names = np.array(['class 1', 'class 2', 'class 3'])
df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75
df['species'] = pd.Categorical.from_codes(target, target_names)
 
train, test = df[df['is_train']==True], df[df['is_train']==False]
features = df.columns[1:14]
#features1 = np.append(df.columns[10:14], df.columns[1])
#features =  np.append(features1, df.columns[7])
 
forest = RFC(n_jobs=2,n_estimators=50)
y, _ = pd.factorize(train['species'])
forest.fit(train[features], y)
 
preds = target_names[forest.predict(test[features])]
print (pd.crosstab(index=test['species'], columns=preds, rownames=['actual'], colnames=['preds']))
importances = forest.feature_importances_
indices = np.argsort(importances)
plt.figure(1)
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Relative Importance')
plt.show()
