import torch
from torch import nn
import matplotlib.pyplot as plt

weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02

X = torch.arange(start,end,step).unsqueeze(dim=1)
y = weight * X + bias

#print(X[:10])
#print("****************************")
#print(y[:10])

#spliting data into training and test
train_split = int(0.8 * len(X))
#x_test =
# y_test =
x_train = X[:train_split]
y_train = y[:train_split]
print(len(x_train),len(y_train))

x_test =  X[train_split:]
y_test =  y[train_split:]
print(len(x_test),len(y_test))

def plot_predictions(train_data=x_train,train_labels=y_train,test_data=x_test,test_labels=y_test,predictions=None):
    plt.figure(figsize=(10,7))
    plt.scatter(train_data,train_labels,c="b",s=4,label="Training data")
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")
    if predictions is not None:
        print("sssssssssssssssssssssssssssssssssssss")
        plt.scatter(test_data, predictions, c="r",s=4, label="predictions")
    plt.legend(prop = {"size":14})
    plt.show()
plot_predictions()




