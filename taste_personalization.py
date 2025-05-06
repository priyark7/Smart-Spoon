
from sklearn.linear_model import LinearRegression
import numpy as np

# Simulated dataset: [age, salt_preference_score, past_response] -> electric stimulation level
X = np.array([
    [25, 7, 1], 
    [40, 5, 0], 
    [30, 6, 1],
    [60, 4, 0]
])
y = np.array([2.5, 1.8, 2.2, 1.5])  # desired stimulation level

model = LinearRegression()
model.fit(X, y)

# Predict stimulation for a new user
new_user = np.array([[35, 6, 1]])
predicted_stimulation = model.predict(new_user)

print(f"Recommended stimulation level: {predicted_stimulation[0]:.2f} volts")
