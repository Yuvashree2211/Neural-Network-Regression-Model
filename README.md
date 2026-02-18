# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Problem Statement

Regression problems aim to predict a continuous numerical value based on input features. Traditional regression models may fail to capture complex non-linear relationships.
A Neural Network Regression Model uses multiple layers of neurons to learn these non-linear patterns and improve prediction accuracy.

## Neural Network Model

<img width="960" height="645" alt="image" src="https://github.com/user-attachments/assets/c4271569-1686-46bd-84c2-e190df2f70f0" />



## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: YUVASHREE R
### Register Number: 212224040378
```python

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
dataset1 = pd.read_csv('Housing.csv')
X = dataset1[['area']].values
y = dataset1[['bathrooms']].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=33)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 8)
        self.fc2 = nn.Linear(8, 10)
        self.fc3 = nn.Linear(10, 1)
        self.relu = nn.ReLU()
        self.history = {'loss': []}

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x) 
        return x
print("Name: YUVASHREE R")
print("Reg No: 212224040378")
yuvas = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(yuvas.parameters(), lr=0.001)
def train_model(yuvas, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(yuvas(X_train), y_train)
        loss.backward()
        optimizer.step()

        yuvas.history['loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')
train_model(yuvas, X_train_tensor, y_train_tensor, criterion, optimizer)
with torch.no_grad():
    test_loss = criterion(yuvas(X_test_tensor), y_test_tensor)
    print(f'Test Loss: {test_loss.item():.6f}')
loss_df = pd.DataFrame(yuvas.history)
import matplotlib.pyplot as plt
loss_df.plot()
print("Name: YUVASHREE R")
print("Reg No: 212224040378")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()

```
## Dataset Information

<img width="271" height="618" alt="image" src="https://github.com/user-attachments/assets/bba4acac-be7c-4bab-abb9-cd5603c42322" />



## OUTPUT
<img width="346" height="285" alt="image" src="https://github.com/user-attachments/assets/e571f706-8330-4929-a8cc-2c0299f19288" />

### Training Loss Vs Iteration Plot

<img width="789" height="633" alt="image" src="https://github.com/user-attachments/assets/61917d4f-faf5-4ab9-b55f-1725cde7a7b8" />




### New Sample Data Prediction

<img width="958" height="221" alt="image" src="https://github.com/user-attachments/assets/15e385be-ffb1-468c-b6a2-4cc2ee9707f8" />



## RESULT

The neural network regression model was successfully developed and trained.
