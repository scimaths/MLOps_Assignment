from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import pandas as pd
import dvc.api
import json
import pickle

# with dvc.api.open(repo="https://github.com/scimaths/MLOps_Assignment.git", path="data/creditcard.csv", mode="r") as fd:
    # df = pd.read_csv(fd)

df = pd.read_csv("../data/creditcard.csv")
df_z = df[df['Class'] == 0]
df_o = df[df['Class'] == 1]
train_z, test_z = train_test_split(df_z, test_size = 0.2)
train_o, test_o = train_test_split(df_o, test_size = 0.2)
train = pd.concat([train_z, train_o])
test = pd.concat([test_z, test_o])
train.to_csv("../data/processed/train.csv")
test.to_csv("../data/processed/test.csv")
X_train = train.iloc[:,:-1].copy()
y_train = train.iloc[:,-1].copy()
X_test = test.iloc[:,:-1].copy()
y_test = test.iloc[:,-1].copy()
clf = DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
f1_score = metrics.f1_score(y_test, y_pred)
print(accuracy, f1_score)
dict_acc = {
    "accuracy": accuracy,
    "f1_weighted_score": f1_score,
}
with open('../metrics/acc_f1.json', 'w') as jsonfile:
    json.dump(dict_acc, jsonfile)
pickle.dump(clf, open('../models/model.pkl', 'wb'))