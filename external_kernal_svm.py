import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train_mine.csv')
X = dataset.iloc[:, 1:6].values
y = dataset.iloc[:, 8].values

dataset_test = pd.read_csv('test_mine.csv')
X_test = dataset_test.iloc[:, 1:6].values

dataset_y = pd.read_csv('gender_submission.csv')
y_test=dataset_y.iloc[:,1].values

#Taking care of missing data
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
X[:,2:3]=imputer.fit_transform(X[:,2:3])

imputer_test=Imputer(missing_values='NaN',strategy='mean',axis=0)
X_test[:,2:3]=imputer_test.fit_transform(X_test[:,2:3])

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_Xtest = LabelEncoder()
X_test[:, 0] = labelencoder_Xtest.fit_transform(X_test[:, 0])

onehotencoder_test = OneHotEncoder(categorical_features = [0])
X_test = onehotencoder_test.fit_transform(X_test).toarray()


# Avoiding the Dummy Variable Trap
X = X[:, 1:]
X_test = X_test[:, 1:]

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
X_test = sc.fit_transform(X_test)

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X, y)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_test, y_pred)

AZero=cm_test[0,0]+cm_test[0,1]
AOne=cm_test[1,0]+cm_test[1,1]
PZero=cm_test[0,0]+cm_test[1,0]
POne=cm_test[0,1]+cm_test[1,1]
x = [0,1]
Y = [AZero,AOne]
x1 = [0,1]
Y1 = [PZero,POne]
plt.bar(x,Y,label="actual",color='red',width=0.2)
plt.bar(x1,Y1,label="predicted",color='green',width=0.2)
plt.title('comparision of actual and predicted values')
plt.ylabel('counts of person')
plt.xlabel('survived or not')
plt.legend()
plt.show()

from sklearn import metrics
print("Mean absolute error: %.2f" % metrics.mean_absolute_error(y_test, y_pred))
print("Mean squared error: %.2f" % metrics.mean_squared_error(y_test, y_pred))
print("Root Mean squared error: %.2f" % np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("accuracy: %.2f" % (1-np.sqrt(metrics.mean_squared_error(y_test, y_pred))))