import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn

#load data set from csv file using PANDAS
data = pd.read_csv("creditcard.csv")

#Exploring the dataset
print(data.columns)

#to know size of data
print(data.shape)

#to check all availability of data
print(data.describe())

#reducing data size foe efficient working
#but this may produce less accuracy in results
data = data.sample(frac = 0.1 ,random_state = 1)
print(data.shape)

#plotting histogram of each module
data.hist(figsize = (20,20))
plt.show()

#determining the no. of fraudcases in data
fraud = data[data["Class"] == 1]
valid = data[data["Class"] == 0]

outlier_fraction = len(fraud)/float(len(valid))

print(outlier_fraction)

print("Fraud transaction - {}".format(len(fraud)))
print("valid transaction - {}".format(len(valid)))

#drawing the correlation matrix
corrmat = data.corr()
fig = plt.figure(figsize = (12,7))
sns.heatmap(corrmat,vmax = 8,square = True)
plt.show()

#get all the coloums from the dataframe
columns = data.columns.tolist()
columns = [c for c in columns if c not in ["Class"]]

#store the variable we will be predicting on
target = "Class"

x = data[columns]
y = data[target]

print(x.shape)
print(y.shape)

from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

#define random state
state = 1

#define outlier detection methods

classifiers = {"Isolation Forest" : IsolationForest(max_samples = len(x),contamination = outlier_fraction,
                                                    random_state = state) ,
               "LocalOutlier Factor" : LocalOutlierFactor(n_neighbors = 20 ,contamination = outlier_fraction)
               }
 
#fit the mode
n_outlier = len(fraud)

for i , (clf_name,clf) in enumerate(classifiers.items()):
    
    #fit the data and tg outliers
    if clf_name == "LocalOutlier Factor":
        y_pred = clf.fit_predict(x)
        score_pred = clf.negative_outlier_factor_
    else:
        clf.fit(x)
        score_pred = clf.decision_function(x)
        y_pred = clf.predict(x)
        
    #reshape the prediction values to o for valid 1 for invalid
    [y_pred[y_pred]==1] == 0
    [y_pred[y_pred]== -1] == 1

    n_errors = (y_pred != y).sum()
    
    #run classification metrics
    print("{}:{}".format(clf_name,n_errors))
    print(accuracy_score(y,y_pred))
    print(classification_report(y,y_pred))
           
     
    
    
    
    
    
    
    
    
    
    