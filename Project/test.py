import numpy as np
from sklearn.linear_model import LinearRegression

# Sample data (replace with your dataset)
# X represents the input features (player statistics)
# y represents the target variable (match outcome)
X = np.array([[x1, x2, x3],   # Features for sample 1
              [x4, x5, x6],   # Features for sample 2
              ...            # Add more samples as needed
             ])
y = np.array([y1, y2, y3, ...])  # Target values for each sample

# Create a linear regression model
model = LinearRegression()

# Fit the model to the data
model.fit(X, y)

# Make predictions for new data
new_data = np.array([[new_x1, new_x2, new_x3]])  # Replace with new input features
predictions = model.predict(new_data)

# Print the predictions
print(predictions)