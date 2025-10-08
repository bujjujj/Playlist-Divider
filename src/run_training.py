import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Load your dataset
df = pd.read_csv('training_features.csv')

# Separate features (all columns except 'label') and the target label
X = df.drop('label', axis=1)
y = df['label']

# Split data into training and testing sets (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training with {len(X_train)} songs, testing with {len(X_test)} songs.")

# Create a pipeline
# 1. SimpleImputer: Handles any missing data points (just in case)
# 2. StandardScaler: Scales all features to have a similar range, which helps many models perform better.
# 3. RandomForestClassifier: The actual machine learning model.
model_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the entire pipeline
print("Training the Random Forest model...")
model_pipeline.fit(X_train, y_train)
print("Training complete!")

# Make predictions on the test set
print("\nEvaluating model on the test set...")
y_pred = model_pipeline.predict(X_test)

# Print evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the final, trained pipeline
joblib.dump(model_pipeline, 'models/song_classifier.joblib')
print("\nFinal model pipeline saved to 'models/song_classifier.joblib'")