
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X = np.array([[500], [600], [700], [800], [900],
              [1000], [1100], [1200], [1300], [1400]])

y = np.array([25, 30, 35, 40, 45, 50, 55, 60, 65, 70])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Predicted Prices:", y_pred)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

new_area = [[1500]]
print("Price for 1500 sq.ft:", model.predict(new_area))

plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')

plt.xlabel('Area (sq.ft)')
plt.ylabel('House Price (Lakhs)')
plt.title('House Price Prediction')
plt.legend()
plt.grid()
plt.show()
