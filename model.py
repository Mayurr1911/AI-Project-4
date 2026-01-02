import pandas as pd
from sklearn.neighbors import NearestNeighbors
import joblib
import numpy as np

# Generate synthetic user-item rating data
np.random.seed(42)
n_users = 100
n_items = 50
ratings = np.random.randint(1, 6, size=(n_users, n_items))
data = pd.DataFrame(ratings, columns=[f'item_{i}' for i in range(n_items)])

# For KNN, use the ratings as features
model = NearestNeighbors(n_neighbors=5, algorithm='ball_tree')
model.fit(data)

# Save model
joblib.dump(model, 'recommendation_model.pkl')
print("Recommendation model trained and saved.")
