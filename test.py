import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score

x = np.array([[500], [600], [700], [800], [900]])
y = np.array([25, 30, 35, 40, 45])

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

model = LinearRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

print("Predicted Prices:",y_pred)
print("Mean Squared Error:",mean_squared_error(y_test,y_pred))
print("R2 Score:",r2_score(y_test,y_pred))

area =[[1000]]
print("Price for 1000 sq.ft:",model.predict(area))
