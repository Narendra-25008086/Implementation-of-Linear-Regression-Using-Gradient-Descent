# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
* Import the required libraries such as NumPy, Pandas, and Matplotlib.
* Load the dataset containing population and profit values of cities.
* Initialize parameters (theta values) and set learning rate and iterations.
* Apply Gradient Descent to minimize the cost function and update parameters.
* Find the best fit regression line after training the model.
* Predict the profit for a given city population using the trained model.
* Plot the dataset points and regression line for visualization.
* Display the predicted output and final parameter values.



## Program:
```
Program to implement the linear regression using gradient descent.
Developed by:NARENDRA KRISHNAN KS
RegisterNumber:212225240096
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def linear_regression (X1, y, learning_rate=0.1, num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta=np.zeros(X.shape[1]).reshape(-1,1)

    for _ in range(num_iters):
        predictions=(X).dot(theta).reshape(-1,1)
        errors= (predictions-y).reshape(-1,1)
    theta -learning_rate*(1/len(X1))*X.T.dot(errors)

    return theta

data=pd.read_csv(r"C:\Introduction to Machine Learning\UNIT 1\Gradient Descent\DATASET-20250226\50_Startups.csv")
data.head(11)

X=(data.iloc[1:,:-2].values)
X1=X.astype(float)

scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)

print("X =",X)

print("X1_Scaled =",X1_Scaled)

theta=linear_regression(X1_Scaled, Y1_Scaled)
new_data= np.array([165349.2, 136897.8, 471784.1]).reshape(-1,1)
new_scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1, new_scaled), theta)
prediction= prediction.reshape(-1,1)
pre = scaler.inverse_transform (prediction)
print("prediction =",prediction)
print(f"Predicted value: {pre}")

## Output:
![linear regression using gradient descent](sam.png)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
