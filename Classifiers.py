from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import math

# Function starts classifier on base logictic regression for vectors of features.
def Logistic_Reression():
    max_epoch = 20
    data = pd.read_csv('Data_vector_reviews.csv')
    X = data.values[::, 1:21]
    y = data.values[::, 0:1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(len(X_train), " train +", len(X_test), "test")

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    for train_index, test_index in split.split(X, y):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    print(data["overall"].value_counts() / len(data))

    lg_clf = LogisticRegression(penalty='l1', solver='liblinear')
    y_train = y_train.ravel()
    lg_clf.fit(X_train, y_train)

    lg_clf_prediction = lg_clf.predict(X_test)
    print("Прогнозы:", lg_clf_prediction)
    print("Метки:", list(y_test))
    print(accuracy_score(lg_clf_prediction, y_test))
    print(metrics.classification_report(y_test, lg_clf_prediction))
    print(metrics.confusion_matrix(y_test, lg_clf_prediction))



# Function starts classifier on base decision tree for vectors of features.
def Decision_Tree():
    data = pd.read_csv('Data_vector_reviews.csv')
    X = data.values[::, 1:21]
    y = data.values[::, 0:1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(len(X_train), " train +", len(X_test), "test")

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    for train_index, test_index in split.split(X, y):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    print(data["overall"].value_counts() / len(data))

    tree_clf = DecisionTreeClassifier()
    tree_clf.fit(X_train, y_train)
    y_pred = tree_clf.predict(X_test)
    # loss = log_loss(y_test, y_pred)
    # print(loss)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


# Function starts classifier on base forest random for vectors of features.
def Forest_random():
    data = pd.read_csv('Data_vector_reviews.csv')
    X = data.values[::, 1:21]
    y = data.values[::, 0:1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(len(X_train), " train +", len(X_test), "test")

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    for train_index, test_index in split.split(X, y):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    print(data["overall"].value_counts() / len(data))

    model = RandomForestClassifier(n_estimators=100, bootstrap=True, max_features='sqrt')
    model.fit(X_train, y_train)

    model_predict = model.predict(X_test)
    print(confusion_matrix(y_test, model_predict))
    print(classification_report(y_test, model_predict))


Logistic_Reression()