
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

X = np.array([
    [1, 1],
    [2, 1],
    [1, 2],
    [2, 2],
    [3, 1],
    [6, 5],
    [7, 6],
    [8, 7],
    [9, 8],
    [10, 9]
])

y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SVC(kernel='linear', C=1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Predictions:", y_pred)
print("Actual Values:", y_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

new_email = scaler.transform([[7, 6]])
print("New Email Classification:", model.predict(new_email))

plt.figure(figsize=(9, 6))

plt.scatter(
    X_train[:, 0],
    X_train[:, 1],
    c=y_train,
    cmap='coolwarm',
    s=80,
    label='Training Data'
)

plt.scatter(
    model.support_vectors_[:, 0],
    model.support_vectors_[:, 1],
    s=180,
    facecolors='none',
    edgecolors='black',
    linewidths=2,
    label='Support Vectors'
)

xx = np.linspace(
    X_train[:, 0].min() - 0.5,
    X_train[:, 0].max() + 0.5,
    100
)

w = model.coef_[0]
b = model.intercept_[0]

yy = -(w[0] * xx + b) / w[1]

yy_upper = -(w[0] * xx + b - 1) / w[1]

yy_lower = -(w[0] * xx + b + 1) / w[1]

plt.plot(xx, yy, 'k-', linewidth=2, label='Decision Boundary')

plt.plot(xx, yy_upper, 'g--', linewidth=2, label='Upper Margin')

plt.plot(xx, yy_lower, 'r--', linewidth=2, label='Lower Margin')

plt.fill_between(
    xx, yy_lower, yy_upper,
    alpha=0.15,
    color='gray',
    label='Margin Region'
)

plt.xlabel('Suspicious Words (Scaled)')
plt.ylabel('Number of Links (Scaled)')
plt.title('SVM Decision Boundary and Margins')

plt.legend()
plt.grid()
plt.show()

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=['Not Spam', 'Spam']
)

disp.plot(cmap='Blues')
plt.title('SVM Confusion Matrix')
plt.show()
