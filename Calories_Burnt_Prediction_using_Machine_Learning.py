#-----------------------------------------------about dataset
#Correlations between calories burnt, exercise and body type
#The calorie is a unit of energy.For historical reasons, two main definitions of "calorie" are in wide use. 
# The large calorie, food calorie, or kilogram calorie was originally defined as the amount of heat needed to raise the temperature of one kilogram of water by one degree Celsius (or one kelvin).
# The small calorie or gram calorie was defined as the amount of heat needed to cause the same increase in one gram of water.
# Thus, 1 large calorie is equal to 1000 small calories.

#------------------------------------------------work flow
#data download
#data pre-processing
#data anlysis
#train test split
#XGBoost Regression

#-------------------------------------------------import useful labrary
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import  train_test_split
from sklearn import metrics
from xgboost import XGBRegressor

#-------------------------------------------------dataset anlysis
data1 = pd.read_csv("C:/Users/kunde/all vs code/ml prject/exercise.csv")
print(data1.shape)
print(data1.columns)
print(data1.head(5))
print(data1.tail(5))
print(data1.info())
print(data1.describe())
print(data1.isnull().sum())
#data = data.drop(columns=["User_ID"], axis=1)
print(data1.head(5))
print(data1["Gender"].value_counts())
data1.replace({"Gender" : {"female" : 0, "male" : 1}}, inplace= True)
print(data1.head(5))
#-------------------------------------------------dataset visulaztion
data2 = pd.read_csv("C:/Users/kunde/all vs code/ml prject/calories.csv")
print(data2.head(5))
data = pd.merge(data1, data2, on="User_ID")
print(data.head(5))
data = data.drop(columns=["User_ID"], axis=1)
print(data.head(5))
#------------------------------------------------dataset separation
y = data["Calories"]
x = data.drop(columns=["Calories"], axis=1)
print(x.head(5))
#-------------------------------------------------dataset train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=2, test_size=0.2)
print(x.shape, x_train.shape, x_test.shape)
print(y.shape, y_train.shape, y_test.shape)
#-------------------------------------------------model selction
model = XGBRegressor()
model.fit(x_train, y_train)
#---------------------------------------------------train data prediction 
y_tr = model.predict(x_train)
accur = metrics.r2_score(y_tr, y_train)
print(accur)
#---------------------------------------------------test data prediction
y_te = model.predict(x_test)
accur = metrics.r2_score(y_te, y_test)
print(accur)
#-------------------------------------------------single data prediction 
a = [0, 20,166.0, 60.0, 14.0, 94.0, 40.3]
arr = np.asarray(a)
arra = arr.reshape(1, -1)
y_pred = model.predict(arra)
print(y_pred, "this is prediction" , "and this is 66.0")

#-----------------------------------------------by use of linerregression 
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)
y_tr = model.predict(x_train)
accur = metrics.r2_score(y_tr, y_train)
print(accur)
y_te = model.predict(x_test)
accur = metrics.r2_score(y_te, y_test)
print(accur)
