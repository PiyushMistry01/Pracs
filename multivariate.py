
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X = np.array([
    [500, 1, 10],
    [600, 2, 8],
    [700, 2, 7],
    [800, 2, 6],
    [900, 3, 5],
    [1000, 3, 4],
    [1100, 3, 3],
    [1200, 4, 2],
    [1300, 4, 1],
    [1400, 4, 1]
])

y = np.array([25, 30, 35, 40, 48, 55, 60, 68, 75, 82])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Predicted Prices:", y_pred)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

new_house = [[1500, 4, 2]]
print("New House Price:", model.predict(new_house))

plt.scatter(y_test, y_pred, color='blue', label='Predicted Values')
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color='red',
    label='Ideal Prediction'
)

plt.xlabel('Actual House Price')
plt.ylabel('Predicted House Price')
plt.title('Actual vs Predicted House Prices')
plt.legend()
plt.grid()
plt.show()
