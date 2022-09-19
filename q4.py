# %% [markdown]
# Importing libraries

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets

# %% [markdown]
# Class Perceptron

# %%
def efficiency(y_pred:np.array, y_actual:np.array)->None:
    eff = np.sum(y_pred==y_actual)*100/len(y_actual)
    return eff


class Perceptron:
    def __init__(self, learning_rate:float, iterations:int, initial_bias=0, initial_weights=None) -> None:
        self.lr = learning_rate
        self.iter = iterations
        self.bias = initial_bias
        self.weights = initial_weights
        pass

    def training(self, xtr_data:np.array, ytr_data:np.array)->np.array:
        nof_samples, nof_features = xtr_data.shape
        self.weights = np.zeros((nof_features, 1))

        #normalizing data
        ytr_data[ytr_data>0] = 1
        ytr_data[ytr_data<=0]=0

        for i in range(self.iter):
            for j in range(nof_samples):

                y_pred = np.dot(self.weights.T, xtr_data[j].reshape((2,1))) + self.bias
                y_pred = [1 if y_pred >0 else 0]

                #provided update rule
                correction = self.lr*(ytr_data[j]- y_pred)*xtr_data[j].reshape((2,1))
                # print(y_pred)
                self.weights+=correction

                #because bias consists of value corresponding to 1 in input data
                self.bias+= self.lr*(ytr_data[j]- y_pred)

        # print(self.weights)

    def validation(self, xval_data, yval_data):
        y_pred = np.dot(self.weights.T, xval_data.T) + self.bias
        y_pred[y_pred>0]=1
        y_pred[y_pred<=0]=0
        return y_pred



# %% [markdown]
# Writing main function to import, train and test the performance of the data

# %%
def main(learning_rate = 100, iter = 1000):
    
    
    #getting th data set from sklearn
    X_data, y_data = datasets.make_blobs(
        n_samples=10100, n_features=2, centers=2, cluster_std=1.05, random_state=2
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.99, random_state=123
    )

    obj = Perceptron(learning_rate, iter)
    obj.training(X_train, y_train)
    y_pred= obj.validation(X_test, y_test)
    print('The performance with learning rate ' + str(learning_rate) + ' is ' + str(efficiency(y_pred, y_test)) + '%')

# %%
if __name__=='__main__':
    learning_r = [100,1,0.01,0.0001]
    for lr in learning_r:
        main(lr)


