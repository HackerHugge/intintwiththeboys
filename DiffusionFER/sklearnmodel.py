import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Load your dataset
current_path = os.getcwd()
csv_file_path = os.path.join(current_path, 'aus_data.csv')
aus_data_df = pd.read_csv(csv_file_path)

# Remove rows with missing values
aus_data_df.dropna(inplace=True)

# Extract features (AUs, and other relevant features) and target variable (valence)
X = aus_data_df.drop(['emotion', 'file', 'valence', 'arousal'], axis=1)
y_valence = aus_data_df['valence']

# Split the data into training and testing sets
X_train, X_test, y_valence_train, y_valence_test = train_test_split(X, y_valence, test_size=0.2, random_state=42)

# Initialize and train the regression model
model = LinearRegression()
model.fit(X_train, y_valence_train)

# Make predictions on the test set
y_valence_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_valence_test, y_valence_pred)
r2 = r2_score(y_valence_test, y_valence_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Save the model to a file
model_file = os.path.join(current_path, "valence_regression_model.joblib")
joblib.dump(model, model_file)

print(f"Model saved to {model_file}")
