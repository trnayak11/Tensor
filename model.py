import torch 
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
#check pytorch version
print(torch.__version__)

#creating device agnostic code
#device = "GPU" if torch.cuda.is_available() else "CPU"
#print(f"defive using = {device}")

#creating data
weights = 0.7
biass = 0.3

start = 0
end = 1
step = 0.02

X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weights * X + biass

#print(X[:10])
#print(y[:10])

#split data for traiing and testing

train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

def plot_predictions(train_data=X_train,train_labels=y_train,test_data=X_test,test_labels=y_test,predictions=None):
    plt.figure(figsize=(10,7))
    plt.scatter(train_data,train_labels,c="b",s=4,label="Training data")
    plt.scatter(test_data,test_labels,c="g",s=4,label="Testing data")
    if predictions is not None:
        plt.scatter(test_data,predictions,c="r",s=4,label="Predictions")
    plt.legend(prop = {"size":14})
    plt.show()

#print(len(X_train),len(y_train),len(X_test),len(y_test))
# plot_predictions(X_train,y_train,X_test,y_test)


#building pytorch linear model
class LinearregressionModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=1,out_features=1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #print("forward function called")
        return self.linear_layer(x)
 
    
torch.manual_seed(42)
model_1 = LinearregressionModelV2()
#print(model_1.state_dict())
#print(model_1(X_train))
test_pred = model_1(X_test)
#training code
# check the device type
# print(next(model_1.parameters()))

#loss function, optimizer, training loop, testing loop

loss_fn = nn.L1Loss()

optimizer = torch.optim.SGD(params = model_1.parameters(),lr=0.01)

#training loop
torch.manual_seed(42)
epochs = 200
for epoch in range(epochs):
    model_1.train()
    y_pred = model_1(X_train)
    #calculate the loss
    loss = loss_fn(y_pred, y_train)
    #optimizer to zer grad
    optimizer.zero_grad()
    #perform back propagation
    loss.backward()
    #optimizer step
    optimizer.step()
    ##testing
    model_1.eval()
    with torch.inference_mode():
        test_pred = model_1(X_test)
        test_loss = loss_fn(test_pred,y_test)
    if epoch % 10 == 0:
        print(f"epoch: {epoch} | Loss: {loss} | Test_loss: {test_loss}")

model_1.eval()
with torch.inference_mode():
    y_test_pred = model_1(X_test)    
    plot_predictions(predictions=y_test_pred)
#save a model
from pathlib import Path
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "01_pytorch_workflow_model_o1.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

print(f"saving model to path {MODEL_PATH}")
torch.save(obj=model_1.state_dict(),f=MODEL_SAVE_PATH)

#LOAD A MODEL
#create new object
loaded_model_1 = LinearregressionModelV2()
loaded_model_1.load_state_dict(torch.load(MODEL_SAVE_PATH))
print(loaded_model_1.state_dict())





