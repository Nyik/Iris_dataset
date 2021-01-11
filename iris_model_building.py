import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

data = pd.read_csv("csv/Iris.csv")

def changeNum(row):
    if row == 'Iris-setosa':
        return 1
    elif row == 'Iris-versicolor':
        return 0
    else:
        return 2

data['Species_Target'] = data['Species'].apply(changeNum)

X = data.drop(columns=['Id','Species_Target','Species'])
Y = data.Species_Target
# st.write(X)
clf = RandomForestClassifier()
clf.fit(X, Y)

pickle.dump(clf, open('iris_clf.pkl', 'wb'))

