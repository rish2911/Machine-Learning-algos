# %% [markdown]
# Importing libraries

# %%
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import array

# %% [markdown]
# 

# %%
training_data = pd.read_excel("mlr05.xls", sheet_name="Mlr05")

# %%
training_data_inp = np.array(training_data)[:20,1:]
test_data_inp = np.array(training_data)[20:,1:]
training_data_labels = np.array(training_data)[:20,0]
test_data_labels = np.array(training_data)[20:,0]



# %%
def raw_r2_score(y_true:array, y_pred:array)->float:
    #sum of square of residuals
    SSR = (np.linalg.norm((y_true - y_pred)))**2
    #sum of square of total
    SST = (np.linalg.norm(y_true-np.mean(y_true)))**2
    return 1 - SSR/SST

# %%
y_true = np.array([1,2,3])
y_pred = np.array([1,2,4])

r2_score(y_true, y_pred)

# %% [markdown]
# Check what r2 score is

# %%
class Simple_Regress:
    def __init__(self, deg = 1):
        #degree will be used for determining type of fit
        self.deg = deg

    def curve_fit(self, X,y):
        samples, features = X.shape

        #just linear for now
        thet_predict = np.linalg.inv(np.dot(X.T, X)).dot(X.T).dot(y)
        return thet_predict

            


          


# %%
try1 = Simple_Regress()
thet = try1.curve_fit(training_data_inp, training_data_labels)

# %%
test_data_labels_pred = (test_data_inp).dot((np.matrix(thet).T))

# %%
test_data_labels_pred

# %%
np.matrix(test_data_labels).shape

# %%
r2 = raw_r2_score(test_data_labels_pred,np.matrix(test_data_labels).T)

print('The r2 score for the developed linear regression model is ',r2 )

