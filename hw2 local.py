import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# Collect data
df = pd.read_csv("CreditCard.csv")

# Data preprocessing
df["Gender"] = df["Gender"].map({"M": 1, "F": 0})
df["CarOwner"] = df["CarOwner"].map({"Y": 1, "N": 0})
df["PropertyOwner"] = df["PropertyOwner"].map({"Y": 1, "N": 0})

# Extract features and target
X = df[['Gender','CarOwner','PropertyOwner','#Children','WorkPhone','Email_ID']].values
y = df['CreditApprove'].values

# Define error function
def e(w, X, y):
    product = np.dot(X, w)
    predictions = np.where(product >= 0, 1, 0)
    error = y - predictions
    return np.sum(error**2)

# Initialize random weights
w = np.random.choice([-1, 1], size=X.shape[1])

# Hill Climbing Local Search Algorithm
error_rates = [e(w, X, y)]

for iteration in range(100):
    improved = False
    for i in range(len(w)):
        w_copy = w.copy()
        w_copy[i] *= -1     # flip ith value
        w_copy_err = e(w_copy, X, y)
        if w_copy_err < error_rates[-1]:
            w = w_copy
            error_rates.append(w_copy_err)
            improved = True
    if not improved:
        break

# Plot the error rate
plt.plot(error_rates)
plt.xlabel("Iteration")
plt.ylabel('Error Rate')
plt.title('Error Rate Across Iterations')
plt.show()

# Output the optimal weights and error rate
print("Optimal weights:", w)
print("Minimum error rate:", error_rates[-1])