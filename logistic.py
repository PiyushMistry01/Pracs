
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

X = np.array([[1], [2], [3], [4], [5],
              [6], [7], [8], [9], [10]])

y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Predictions:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

hours = [[7]]
print("Result for 7 hours:", model.predict(hours))

plt.scatter(X, y, color='blue', label='Actual Data')

x_values = np.linspace(0, 11, 200).reshape(-1, 1)
y_prob = model.predict_proba(x_values)[:, 1]

plt.plot(x_values, y_prob, color='red', label='Logistic Curve')
plt.axhline(y=0.5, color='green', linestyle='--', label='Decision Threshold')

plt.xlabel('Study Hours')
plt.ylabel('Probability of Passing')
plt.title('Student Pass/Fail Prediction')
plt.legend()
plt.grid()
plt.show()
