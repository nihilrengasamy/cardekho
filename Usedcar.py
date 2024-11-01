import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.linear_model import LinearRegression

# reading the dataset
cars = pd.read_csv("whitewolfcar.csv")

print(cars.info())

print(cars.head())

print(cars['symboling'].astype('category').value_counts())

# aspiration: An (internal combustion) engine property showing 
# whether the oxygen intake is through standard (atmospheric pressure)
# or through turbocharging (pressurised oxygen intake)

print(cars['aspiration'].astype('category').value_counts())

print(cars['drivewheel'].astype('category').value_counts())

# all numeric (float and int) variables in the dataset
cars_numeric = cars.select_dtypes(include=['float64', 'int'])
print(cars_numeric.head())

# dropping symboling and car_ID 
cars_numeric = cars_numeric.drop(['symboling', 'car_ID'], axis=1)
print(cars_numeric.head())

# correlation matrix
cor = cars_numeric.corr()
print(cor)

# variable formats
print(cars.info())

# converting symboling to categorical
cars['symboling'] = cars['symboling'].astype('object')
print(cars.info())

# CarName: first few entries
print(cars['CarName'][:30])

# Extracting carname

# Method 1: str.split() by space
carnames = cars['CarName'].apply(lambda x: x.split(" ")[0])
print(carnames[:30])

# Method 2: Use regular expressions
import re

# regex: any alphanumeric sequence before a space, may contain a hyphen
p = re.compile(r'\w+-?\w+')
carnames = cars['CarName'].apply(lambda x: re.findall(p, x)[0])
print(carnames)

# New column car_company
cars['car_company'] = cars['CarName'].apply(lambda x: re.findall(p, x)[0])

print(# look at all values 
cars['car_company'].astype('category').value_counts())

# replacing misspelled car_company names

# volkswagen
cars.loc[(cars['car_company'] == "vw") | 
         (cars['car_company'] == "vokswagen")
         , 'car_company'] = 'volkswagen'

# porsche
cars.loc[cars['car_company'] == "porcshce", 'car_company'] = 'porsche'

# toyota
cars.loc[cars['car_company'] == "toyouta", 'car_company'] = 'toyota'

# nissan
cars.loc[cars['car_company'] == "Nissan", 'car_company'] = 'nissan'

# mazda
cars.loc[cars['car_company'] == "maxda", 'car_company'] = 'mazda'

print(cars['car_company'].astype('category').value_counts())

# drop carname variable
cars = cars.drop('CarName', axis=1)

print(cars.info())

# outliers
print(cars.describe())

print(cars.info())

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

cars['fueltype'] = label_encoder.fit_transform(cars['fueltype'])
cars['doornumber'] = label_encoder.fit_transform(cars['doornumber'])
cars['carbody'] = label_encoder.fit_transform(cars['carbody'])
cars['drivewheel'] = label_encoder.fit_transform(cars['drivewheel'])
cars['enginelocation'] = label_encoder.fit_transform(cars['enginelocation'])
cars['enginetype'] = label_encoder.fit_transform(cars['enginetype'])


cars = cars[['fueltype','doornumber','carbody','drivewheel','enginelocation','carlength','carwidth','carheight','curbweight','enginetype','horsepower','peakrpm','price']]
print(cars)

from sklearn.preprocessing import LabelEncoder

# Drop features with high correlation (> 0.9) or low correlation with the target variable
correlation_matrix = cars.corr()
g = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
print(g)

threshold = 0.95
high_corr_features = [column for column in correlation_matrix.columns if abs(correlation_matrix[column].corr(cars['price'])) < threshold]
cars = cars.drop(columns=high_corr_features)
print(cars)

X = cars.drop(columns=['price'])
y = cars['price']

print(X)
print(y)

"""
# creating dummy variables for categorical variables

# subset all categorical variables
cars_categorical = X.select_dtypes(include=['object'])
print(cars_categorical.head())

# convert into dummies
cars_dummies = pd.get_dummies(cars_categorical, drop_first=True)
print(cars_dummies.head())

# drop categorical variables 
X = X.drop(list(cars_categorical.columns), axis=1)

# concat dummy variables with X
X = pd.concat([X, cars_dummies], axis=1)
"""
# scaling the features
from sklearn.preprocessing import scale

# storing column names in cols, since column names are (annoyingly) lost after 
# scaling (the df is converted to a numpy array)
cols = X.columns
X = pd.DataFrame(scale(X))
X.columns = cols
print(X.columns)


# split into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    train_size=0.7,
                                                    test_size = 0.3, random_state=100)

# Building the first model with all the features

# instantiate
lm = LinearRegression()

# fit
lm = lm.fit(X_train, y_train)

# print coefficients and intercept
print(lm.coef_)
print(lm.intercept_)

# predict 
y_pred = lm.predict(X_test)

# metrics
from sklearn.metrics import r2_score

print("The final R2 score is :",r2_score(y_true=y_test, y_pred=y_pred))

#Model Creation for Flask

import pickle

# Save the model
with open('classifier.pkl', 'wb') as file:
    pickle.dump(lm, file)

# Load the model
with open('classifier.pkl', 'rb') as file:
    model_final = pickle.load(file)
print(model_final)