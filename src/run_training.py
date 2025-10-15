# In src/run_training.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def train_model():
    """
    Loads the feature data, trains a classifier, evaluates it,
    and saves the final model.
    """
    # --- 1. Load and Prepare Data ---
    print("Loading feature dataset...")
    try:
        df = pd.read_csv('training_features.csv')
    except FileNotFoundError:
        print("Error: 'training_features.csv' not found. Please run the data gathering script first.")
        return

    print(f"Loaded {len(df)} songs across {len(df['label'].unique())} playlists.")

    # Drop non-feature columns. We keep 'artist' and 'track' for now if they exist,
    # but will drop them before training.
    X = df.drop(['label', 'artist', 'track'], axis=1, errors='ignore')
    y = df['label']

    # --- 2. Split Data into Training and Testing Sets ---
    # `stratify=y` is important for imbalanced datasets. It ensures the test set
    # has the same proportion of songs from each playlist as the training set.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training with {len(X_train)} songs, testing with {len(X_test)} songs.")

    # --- 3. Build the Model Pipeline ---
    # We'll use a RandomForestClassifier.
    # `class_weight='balanced'` is the key to handling your imbalanced playlists!
    # It tells the model to give more weight to songs from smaller playlists during training.
    model_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
    ])

    # --- 4. Train and Evaluate the Model ---
    print("\nTraining the Random Forest model...")
    model_pipeline.fit(X_train, y_train)
    print("Training complete!")

    print("\nEvaluating model on the unseen test set...")
    y_pred = model_pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nOverall Model Accuracy: {accuracy * 100:.2f}%")

    # The Classification Report is the most important output here!
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))

    # --- 5. Save the Final Model ---
    print("\nSaving the final trained model...")
    joblib.dump(model_pipeline, 'models/song_classifier.joblib')
    print("Final model pipeline saved to 'models/song_classifier.joblib'")
    print("You are now ready to use 'classify_playlist.py'!")

    # Generate the confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=model_pipeline.classes_)

    # Plot the confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=model_pipeline.classes_, yticklabels=model_pipeline.classes_)
    plt.ylabel('Actual Playlist')
    plt.xlabel('Predicted Playlist')
    plt.title('Confusion Matrix')
    plt.show()


if __name__ == '__main__':
    train_model()