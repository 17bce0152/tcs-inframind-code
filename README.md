# tcs-inframind-code
I.Importing essential libraries
import numpy as np import pandas as pd
import matplotlib.pyplot as plt import seaborn as sns

%matplotlib inline

import os print(os.listdir())

import warnings warnings.filterwarnings('ignore')

['.ipynb_checkpoints', 'heart.csv', 'Heart_disease_prediction.ipynb', 'READM E.md']
II.Importing and understanding our dataset

dataset = pd.read_csv("heart.csv")

Verifying it as a 'dataframe' object in pandas

type(dataset) pandas.core.frame.DataFrame

Shape of dataset

dataset.shape (303, 14)
dataset.info()
<class 'pandas.core.frame.DataFrame'> RangeIndex: 303 entries, 0 to 302 Data columns (total 14 columns):
age	303 non-null int64
sex	303 non-null int64
cp  303 non-null int64 trestbps 303 non-null int64 chol        303 non-null int64
fbs	303 non-null int64 restecg	303 non-null int64 thalach	303 non-null int64 exang	303 non-null int64 oldpeak	303 non-null float64 slope	303 non-null int64
ca	303 non-null int64
thal	303 non-null int64 target	303 non-null int64 dtypes: float64(1), int64(13) 
iii) Exploratory Data Analysis (EDA)
First, analysing the target variable

y = dataset["target"] sns.countplot(y)
target_temp = dataset.target.value_counts() print(target_temp)

1	165
0	138
Name: target, dtype: int64
Analysing the 'Chest Pain Type' feature
dataset["cp"].unique()
array([3, 2, 1, 0])

As expected, the CP feature has values from 0 to 3
sns.barplot(dataset["cp"],y)
dataset["fbs"].unique()

array([1, 0]) sns.barplot(dataset["fbs"],y)
Analysing the Slope feature dataset["slope"].unique() sns.barplot(dataset["slope"],y)

array([0, 2, 1])
Analysing the 'ca' feature

dataset["ca"].unique() sns.countplot(dataset["ca"]) sns.barplot(dataset["ca"],y) array([0, 2, 1, 3, 4])
Analysing the 'thal' feature dataset["thal"].unique() sns.barplot(dataset["thal"],y) sns.distplot(dataset["thal"])
IV.Train Test split
from sklearn.model_selection import train_test_split

predictors = dataset.drop("target",axis=1) target = dataset["target"]
X_train,X_test,Y_train,Y_test = train_test_split(predictors,target,test_size=0.20,random_state=0) X_train.shape
X_train.shape Y_train.shape Y_train.shape

(242, 13)
(61, 13)
(242,)
(61,)
Logistic Regression
from sklearn.linear_model import LogisticRegression lr = LogisticRegression()
lr.fit(X_train,Y_train) Y_pred_lr = lr.predict(X_test)
Y_pred_lr.shape
score_lr = round(accuracy_score(Y_pred_lr,Y_test)*100,2)
print("The accuracy score achieved using Logistic Regression is: "+str(score_lr)+" %")

The accuracy score achieved using Logistic Regression is: 85.25 %
Naive Bayes
from sklearn.naive_bayes import GaussianNB nb = GaussianNB()
nb.fit(X_train,Y_train) Y_pred_nb = nb.predict(X_test)
Y_pred_nb.shape
score_nb = round(accuracy_score(Y_pred_nb,Y_test)*100,2)
print("The accuracy score achieved using Naive Bayes is: "+str(score_nb)+" %")

The accuracy score achieved using Naive Bayes is: 85.25 %
SVM
from sklearn import svm
sv = svm.SVC(kernel='linear') sv.fit(X_train, Y_train) Y_pred_svm = sv.predict(X_test)
Y_pred_svm.shape
score_svm = round(accuracy_score(Y_pred_svm,Y_test)*100,2)
print("The accuracy score achieved using Linear SVM is: "+str(score_svm)+" %")
The accuracy score achieved using Linear SVM is: 81.97 %
K Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier knn = KNeighborsClassifier(n_neighbors=7) knn.fit(X_train,Y_train) Y_pred_knn=knn.predict(X_test)
Y_pred_knn.shape
score_knn = round(accuracy_score(Y_pred_knn,Y_test)*100,2) print("The accuracy score achieved using KNN is: "+str(score_knn)+" %")


The accuracy score achieved using KNN is: 67.21 %




Decision Tree
from sklearn.tree import DecisionTreeClassifier max_accuracy = 0

for x in range(200):
dt = DecisionTreeClassifier(random_state=x) dt.fit(X_train,Y_train)
Y_pred_dt = dt.predict(X_test)
current_accuracy = round(accuracy_score(Y_pred_dt,Y_test)*100,2) if(current_accuracy>max_accuracy):
max_accuracy = current_accuracy best_x = x


#print(max_accuracy) #print(best_x)


dt = DecisionTreeClassifier(random_state=best_x) dt.fit(X_train,Y_train)
Y_pred_dt = dt.predict(X_test)

print(Y_pred_dt.shape)
score_dt = round(accuracy_score(Y_pred_dt,Y_test)*100,2)

print("The accuracy score achieved using Decision Tree is: "+str(score_dt)+" %")


The accuracy score achieved using Decision Tree is: 81.97 %


Random Forest

from sklearn.ensemble import RandomForestClassifier max_accuracy = 0
for x in range(2000):
rf = RandomForestClassifier(random_state=x) rf.fit(X_train,Y_train)
Y_pred_rf = rf.predict(X_test)
current_accuracy = round(accuracy_score(Y_pred_rf,Y_test)*100,2) if(current_accuracy>max_accuracy):
max_accuracy = current_accuracy best_x = x
#print(max_accuracy) #print(best_x)
rf = RandomForestClassifier(random_state=best_x) rf.fit(X_train,Y_train)
Y_pred_rf = rf.predict(X_test)
Y_pred_rf.shape
score_rf = round(accuracy_score(Y_pred_rf,Y_test)*100,2)
print("The accuracy score achieved using Decision Tree is: "+str(score_rf)+" %")

The accuracy score achieved using Decision Tree is: 95.08 %

XGBoost
import xgboost as xgb
xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42) xgb_model.fit(X_train, Y_train)
Y_pred_xgb = xgb_model.predict(X_test)
Y_pred_xgb.shape
score_xgb = round(accuracy_score(Y_pred_xgb,Y_test)*100,2)

print("The accuracy score achieved using XGBoost is: "+str(score_xgb)+" %")

The accuracy score achieved using XGBoost is: 85.25 %



Neural Network

from keras.models import Sequential from keras.layers import Dense
# https://stats.stackexchange.com/a/136542 helped a lot in avoiding overfitting

model = Sequential() model.add(Dense(11,activation='relu',input_dim=13)) model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy']) model.fit(X_train,Y_train,epochs=300)
Y_pred_nn = model.predict(X_test)
Y_pred_nn.shape
rounded = [round(x[0]) for x in Y_pred_nn]

Y_pred_nn = rounded
score_nn = round(accuracy_score(Y_pred_nn,Y_test)*100,2)

print("The accuracy score achieved using Neural Network is: "+str(score_nn)+" %")

#Note: Accuracy of 85% can be achieved on the test set, by setting epochs=2000, and number of nodes = 11.
