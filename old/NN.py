import torch
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt


device = 'mps' if torch.backends.mps.is_available() else 'cpu'

iris = load_iris()
x, y = iris.data, iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)
x_tr_tensor = torch.tensor(x_train, dtype=torch.float32)
y_tr_tensor = torch.tensor(y_train, dtype=torch.long)

class FullyConnected(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(4, 64)
        self.act1 = nn.ReLU()
        self.l2 = nn.Linear(64, 16)
        self.act2 = nn.ReLU()
        self.l3 = nn.Linear(16, 3)
    
    def forward(self, x):
        x = self.l1(x)
        x = self.act1(x)
        x = self.l2(x)
        x = self.act2(x)
        x = self.l3(x)
        return x

    def fit(self, x, y, epochs=3000, lr=0.002):
        loss_arr = []
        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):
            y_prediction = self(x)
            loss = loss_function(y_prediction, y)
            loss_arr.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        plt.plot(loss_arr)
        plt.show()

model = FullyConnected().to(device)
x_tr_tensor = x_tr_tensor.to(device)
y_tr_tensor = y_tr_tensor.to(device)
model.fit(x_tr_tensor, y_tr_tensor)

x_ts_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
model.eval()
with torch.no_grad():
    y_test_pred = model(x_ts_tensor)
newytest = torch.argmax(y_test_pred, dim=1).cpu().numpy()

print("Accuracy:", accuracy_score(y_test, newytest))
print("Confusion Matrix:\n", confusion_matrix(y_test, newytest))