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

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier.fit(X, y)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

"""
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, y_pred)
"""

from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_test, y_pred)

"""
df=pd.DataFrame(data=X,columns=['Sex','pclass','Age','SibSp','parch'])
#df_y=pd.DataDrame(data=y_pred)
#dataset_y = pd.read_csv('gender_submission.csv')

df['Age'].plot(color='red')
df['pclass'].plot(color='pink')
df['Sex'].plot(color='blue')
df['SibSp'].plot(color='yellow')
df['parch'].plot(color='green')

plt.scatter(df['Age'],df['Sex'],df['pclass'],df['parch'])
#plt.scatter(df['Age'],df['parch'],df['SibSp'],df['Sex'],df['pclass'])
"""
""" not working....

df['Age','Sex'].scatter()
df['pclass'].scatter()
df['Sex'].scatter()
df['SibSp'].scatter()
df['parch'].scatter()
plt['Age'].scatter()
plt['Age','Sex'].scatter()
"""
"""

from sklearn.preprocessing import StandardScaler
sc_y = StandardScaler()
y=sc_y.fit_transform(y)
y_pred=sc_y.fit_transform(y_pred)
"""

"""
#for y vs. y_pred...not complete graph and y and y_pred n feature scaling ma kaik msg aave 6 ana lidhe output nai avtu
plt.scatter(y_hat, y_pred, color = 'red')
plt.plot(y_hat, y_pred, color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
"""
#plt.scatter(y,y_pred) this is for every algo graph showing but overlap j thase cz feature scaling nai thatu so....
"""
plt.plot(X,y)
plt.show()

#plt.scatter(X, y, color = 'red')
plt.scatter(X, y)
plt.plot(X, y)
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


plt.scatter(X,classifier.predict(X),color='red')
plt.plot(X,classifier.predict(X))
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(X, y)#video na training set ni jem 
plt.plot(X,classifier.predict(X))
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()"""

"""
from matplotlib.colors import ListedColormap
X_set, y_set = X, y
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('K-NN (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

plt.plot(X,y)
plt.plot(X_test,y_pred)

plt.plot(X,y)
plt.plot(X_test,y_test)

x = [[0,1],[0,1]]
Y = [[266,152],[262,156]]
plt.bar(x,Y)

x = [0,0,1,1]
Y = [266,262,152,156]
plt.bar(x,Y)

x = [0,1]
Y = [262,156]
plt.bar(x,Y,label="predicted")
plt.title('Info')
plt.ylabel('Y axis')
plt.xlabel('X axis')
plt.legend()
plt.show()
"""
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
"""
this is used to plot all different types like bar or scatter or plot in same graph
or for same type of values comarision via different colors like here we want to use
for prediction vs. count and actual vs. count in same graph and comparision of it so
 we can use this common execution technique.....baki eklu bar vadu run karis to b 
 graph to bane j 6 pn be bar n same graph mate common(jode) execution of that 2 lines
"""
 
from sklearn import metrics
print("Mean absolute error: %.2f" % metrics.mean_absolute_error(y_test, y_pred))
print("Mean squared error: %.2f" % metrics.mean_squared_error(y_test, y_pred))
print("Root Mean squared error: %.2f" % np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("accuracy: %.2f" % (1-np.sqrt(metrics.mean_squared_error(y_test, y_pred))))