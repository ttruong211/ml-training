import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score 


# import the dataset
training = pd.read_csv("train.csv")
testing = pd.read_csv('test.csv')

# number of records in the training set
records = len(training)
print("Number of records: %s " % records)

# mean value of the price
mean_price = training['Price'].mean()
print("Mean value of the price: %s " % mean_price)

# minimal price
min_price = training['Price'].min()
print("Minimum price: %s " % min_price)

# maximal price
max_price = training['Price'].max()
print("Maximum price: %s " % max_price)

# standard derivation of the price
std_price = training['Price'].std()
print("Standard deviation of the price: %s " % std_price)

# plot histogram of the price
plt.hist(training['Price'])
plt.title('Histogram of the price')
plt.ylabel('Occurrences')
plt.xlabel('Price')
plt.show()


# select the specified features
selected_features = ['GrLivArea', 'BedroomAbvGr', 'TotalBsmtSF', 'FullBath']

# create pair-wise scatter plots
sns.pairplot(training[selected_features], height=1.5)
plt.show()

# calculates the predicted value of the price from current weights and feature values.

def pred(features, weights):
    pred_price = np.dot(features, weights)
    return pred_price

# calculates the loss from predicted sale price and the correct sale price

def loss(predPrice, correctPrice):
    mse = np.mean((predPrice - correctPrice)**2)
    return mse

# calculates the gradient of loss function

def gradient(weights, features, correctPrice):
    pred_price = np.dot(features, weights)  # predicted price
    gradient = (2/len(correctPrice)) * \
        np.dot(np.transpose(features), pred_price - correctPrice)
    return gradient

# update weights based on gradient

def update(weights, a, gradient):
    return weights - (a*gradient)


def trainModel(a1, a2, interations, features, correctPrice, weights):
    min_value = float('inf')
    weight1 = weights
    weight2 = weights

    # array contains MSE values through each interations
    MSE1 = []
    MSE2 = []

    # run through loop of interations
    for i in range(interations):
        # calculates the predicted values
        predict1 = pred(features, weight1)
        predict2 = pred(features, weight2)

        # calculates the mse from predicted values and add to the two arrays
        mse_value1 = loss(predict1, correctPrice)
        MSE1.append(mse_value1)
        mse_value2 = loss(predict2, correctPrice)
        MSE2.append(mse_value2) 

        # calculates the gradient of loss function
        grad1 = gradient(weight1, features, correctPrice)
        grad2 = gradient(weight2, features, correctPrice)

        # update weights based on gradient
        weight1 = update(weight1, a1, grad1)
        weight2 = update(weight2, a2, grad2)

        print("Predicted Value (1) for %s iterations - %s" % (i, predict1))
        print("Predicted Value (2) %s iterations - %s" % (i, predict2)) 

    # plotting
    plt.plot(MSE1)
    plt.plot(MSE2)
    plt.title('Training Model: Learning curve in different alphas')
    plt.ylabel('MSE')
    plt.xlabel('Iterations')
    plt.legend(["Learning rate: %s" % a1, "Learning rate: %s" % a2])
    plt.show()

# test the training


# set alpha values
a1 = 8e-10
a2 = 1e-9

# number of iterations
iterations = 500

# generates features from training and test data
features = training.iloc[:, 1:-1].values
features_test = testing.iloc[:, 1:-1].values

# generates price column from training and test data
correctPrice = training['Price'].values
correctPrice_test = testing['Price'].values

# generates random weight vector
weights = np.random.rand(features.shape[1])
# training
trainModel(a1, a2, iterations, features, correctPrice, weights)






