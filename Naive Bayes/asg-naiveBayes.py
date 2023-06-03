
import pandas as pd

test = pd.read_csv("SalaryData_Test.csv")
test

train = pd.read_csv("SalaryData_Train.csv")
train


test_cont = test[test.columns[[0,3,9,10,11]]]
test_cat = test[test.columns[[1,2,4,5,6,7,8,12,13]]]

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()

for i in range(0,9):
    test_cat.iloc[:,i] = LE.fit_transform(test_cat.iloc[:,i])

y_test = test_cat["Salary"]
y_test

test_cat = test_cat.iloc[:,0:8]
test_cat
test_cont

x_test = pd.concat([test_cat,test_cont],axis=1)
x_test
#============================================================================
#TRAIN
#============================================================================

train_cont = train[train.columns[[0,3,9,10,11]]]
train_cat = train[train.columns[[1,2,4,5,6,7,8,12,13]]]

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()

for i in range(0,9):
    train_cat.iloc[:,i] = LE.fit_transform(train_cat.iloc[:,i])


y_train = train_cat["Salary"]
y_train

train_cat = train_cat.iloc[:,0:8]
train_cat
train_cont


x_train = pd.concat([train_cont,train_cat],axis=1)

x_train
y_train
x_test
y_test
from sklearn.naive_bayes import MultinomialNB

NB = MultinomialNB()

NB.fit(x_train, y_train)

y_pred_train = NB.predict(x_train)
y_pred_test = NB.predict(x_test)

y_pred_test
y_pred_train

from sklearn.metrics import accuracy_score

print("Train Accuracy :",accuracy_score(y_train, y_pred_train))
print("Test accuracy : ",accuracy_score(y_test, y_pred_test))

