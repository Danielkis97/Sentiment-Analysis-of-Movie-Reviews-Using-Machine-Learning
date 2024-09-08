import subprocess
import sys
import joblib  # Import joblib for saving and loading models
from load_data_big import prepare_dataset as prepare_large_dataset
from sklearn.model_selection import GridSearchCV

# Ensure scikit-learn is installed
try:
    import sklearn
except ImportError:
    print("scikit-learn not found, installing it now...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])

# Ensure nltk is installed
try:
    import nltk
except ImportError:
    print("nltk not found, installing it now...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "nltk"])

# Ensure pandas is installed
try:
    import pandas as pd
except ImportError:
    print("pandas not found, installing it now...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])

# Ensure matplotlib and seaborn are installed (optional for plots)
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not found, installing it now...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])

try:
    import seaborn as sns
except ImportError:
    print("seaborn not found, installing it now...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "seaborn"])

# Import other necessary libraries
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from load_data_small import prepare_small_dataset
from load_data_big import prepare_data

def vectorize_data(X_train, X_val, X_test):
    vectorizer = CountVectorizer(max_features=10000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)
    return X_train_vec, X_val_vec, X_test_vec, vectorizer

def train_model(X_train_vec, y_train):
    # Use GridSearchCV to find the best alpha
    alpha_values = [0.3, 0.6, 0.9, 1.2, 1.5, 1.8]
    param_grid = {'alpha': alpha_values}
    nb_model = MultinomialNB()
    grid_search = GridSearchCV(estimator=nb_model, param_grid=param_grid, cv=10, scoring='accuracy')
    grid_search.fit(X_train_vec, y_train)

    # Return the best model found by GridSearchCV
    best_model = grid_search.best_estimator_
    print(f"Best Hyperparameter (Alpha): {grid_search.best_params_['alpha']}")
    print(f"Best Cross-Validation Accuracy: {grid_search.best_score_}")
    return best_model, grid_search.best_score_

def evaluate_model(model, X_vec, y_true, dataset_type="Test"):
    y_pred = model.predict(X_vec)
    accuracy = accuracy_score(y_true, y_pred)

    # Get classification report as a dict to extract metrics
    report = classification_report(y_true, y_pred, target_names=['negative', 'positive'], output_dict=True)

    # Confusion matrix
    matrix = confusion_matrix(y_true, y_pred)
    print(f"\nEvaluation on {dataset_type} Set:")
    print("Confusion Matrix:")
    print(matrix)

    # Detailed Confusion Matrix Explanation
    print(f"\nDetailed Confusion Matrix Explanation for {dataset_type} Set:")
    print("True Negatives (correctly predicted negatives):", matrix[0][0])
    print("False Positives (incorrectly predicted positives):", matrix[0][1])
    print("False Negatives (incorrectly predicted negatives):", matrix[1][0])
    print("True Positives (correctly predicted positives):", matrix[1][1])

    # Extract and print Precision, Recall, F1-Score for both classes
    precision_neg = report['negative']['precision']
    recall_neg = report['negative']['recall']
    f1_neg = report['negative']['f1-score']

    precision_pos = report['positive']['precision']
    recall_pos = report['positive']['recall']
    f1_pos = report['positive']['f1-score']

    print(
        f"\nPrecision (Negative): {precision_neg:.2f}, Recall (Negative): {recall_neg:.2f}, F1-Score (Negative): {f1_neg:.2f}")
    print(
        f"Precision (Positive): {precision_pos:.2f}, Recall (Positive): {recall_pos:.2f}, F1-Score (Positive): {f1_pos:.2f}")

    return accuracy, report

if __name__ == "__main__":
    # Prompt user to choose small or large dataset
    dataset_choice = input("Choose dataset (small/large): ").strip().lower()

    # Set unique file names based on dataset size
    if dataset_choice == "small":
        prepare_dataset = prepare_small_dataset
        model_filename = "naive_bayes_model_small.pkl"
        vectorizer_filename = "tfidf_vectorizer_small.pkl"
    elif dataset_choice == "large":
        prepare_dataset = prepare_large_dataset
        model_filename = "naive_bayes_model_large.pkl"
        vectorizer_filename = "tfidf_vectorizer_large.pkl"
    else:
        print("Invalid choice. Please choose 'small' or 'large'.")
        sys.exit(1)

    # Load the dataset with train, validation, and test splits
    X_train, X_val, X_test, y_train, y_val, y_test, original_train, original_val, original_test = prepare_dataset()

    if X_train is not None:
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")

    # Assuming original reviews are needed for later comparison
    original_reviews = original_test

    if X_train is not None:
        # Vectorize the train, validation, and test data
        X_train_vec, X_val_vec, X_test_vec, vectorizer = vectorize_data(X_train, X_val, X_test)

        # Train the model with hyperparameter tuning (GridSearchCV for alpha)
        model, best_cv_accuracy = train_model(X_train_vec, y_train)

        # Save the model, vectorizer, and test data with "latest" filenames
        joblib.dump(model, "latest_model.pkl")
        joblib.dump(vectorizer, "latest_vectorizer.pkl")
        joblib.dump((X_test, y_test, original_reviews), "latest_test_data.pkl")  # Save the test data and original reviews
        print(f"Model and vectorizer saved as 'latest_model.pkl' and 'latest_vectorizer.pkl' successfully.")

        # Evaluate the model on the validation set first
        evaluate_model(model, X_val_vec, y_val, dataset_type="Validation")

        # Evaluate the model on the test set
        evaluate_model(model, X_test_vec, y_test, dataset_type="Test")

        print(f"\nBest Cross-Validation Accuracy: {best_cv_accuracy}")
