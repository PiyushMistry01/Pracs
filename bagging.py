
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

X = np.array([
    [22, 20000],
    [25, 25000],
    [28, 30000],
    [30, 35000],
    [35, 40000],
    [40, 45000],
    [45, 50000],
    [50, 55000],
    [55, 60000],
    [60, 65000]
])

y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=10,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Predictions:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

new_customer = [[32, 38000]]
print("New Customer Prediction:", model.predict(new_customer))

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=['No Purchase', 'Purchase']
)

disp.plot(cmap='Blues')
plt.title('Bagging Confusion Matrix')
plt.show()

plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual')
plt.scatter(range(len(y_pred)), y_pred, color='red', marker='x', label='Predicted')

plt.xlabel('Test Data')
plt.ylabel('Purchase Status')
plt.title('Bagging: Actual vs Predicted')
plt.yticks([0, 1], ['No Purchase', 'Purchase'])
plt.legend()
plt.grid()
plt.show()
