import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02

X = torch.arange(start,end,step).unsqueeze(dim=1)
y = weight * X + bias

train_split = int(0.8 * len(X))
x_train = X[:train_split]
y_train = y[:train_split]
#print(len(x_train),len(y_train))

x_test =  X[train_split:]
y_test =  y[train_split:]
#print(len(x_test),len(y_test))

def plot_predictions(train_data=x_train,train_labels=y_train,test_data=x_test,test_labels=y_test,predictions=None):
    plt.figure(figsize=(10,7))
    plt.scatter(train_data,train_labels,c="b",s=4,label="Training data")
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")
    if predictions is not None:
        plt.scatter(test_data, predictions, c="r",s=4, label="predictions")
    plt.legend(prop = {"size":14})
    plt.show()
#plot_predictions()


class LinearRegressionModel(nn.Module):
    def __init__(self):
        print("init called")
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, requires_grad=True,dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1,requires_grad=True,dtype=torch.float))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #print("foward fun called")
        return self.weights * x + self.bias
    def show(self):
        print(f"weights = {self.weights}")
        print(f"bias = {self.bias}")
torch.manual_seed(42)
model_0 = LinearRegressionModel()
print(model_0.state_dict())


loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_0.parameters(),lr=0.01)
epoch_count = []
train_loss_values = []
test_loss_values = []
epochs = 200
for epoch in range(epochs):
    model_0.train()  
    y_pred = model_0(x_train)
    loss = loss_fn(y_pred,y_train)
    #print(f"Loss :- {loss}")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model_0.eval()
    with torch.inference_mode():
        y_test_pred = model_0(x_test)
        test_loss = loss_fn(y_test_pred,y_test)
    if epoch % 10 == 0:
        epoch_count.append(epoch)
        train_loss_values.append(loss)
        test_loss_values.append(test_loss)
        print(f"Epoch = {epoch} | loss = {loss} | test_loss = {test_loss}")

plt.plot(epoch_count,np.array(torch.tensor(train_loss_values).numpy()),label="Train Loss")
plt.plot(epoch_count,test_loss_values,label="Test Loss")
plt.title("training and test loss curv")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.show()














