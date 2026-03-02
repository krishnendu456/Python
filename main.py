import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression

# Load iris dataset
iris = load_iris()

# Convert to DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Select feature (X) and target (y)
X = df[['sepal length (cm)']]
y = df['petal length (cm)']

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Predict
y_pred = model.predict(X)

# Print results
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

# Plot
plt.scatter(X, y)
plt.plot(X, y_pred,color='red')
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.title("Linear Regression on Iris Dataset")
plt.show()