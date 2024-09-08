import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from data_preprocessing import preprocess_steps  # Use full preprocessing
import random


# Explanation of sentiment probabilities with percentages
def explain_sentiment_probabilities(sentiment_prob):
    negative_prob = sentiment_prob[0][0] * 100
    positive_prob = sentiment_prob[0][1] * 100

    return (f"Prediction Probabilities: Negative: {negative_prob:.2f}%, "
            f"Positive: {positive_prob:.2f}%\n"
            f"Based on these probabilities, the model predicts that this review is more likely to be "
            f"{'negative' if negative_prob > positive_prob else 'positive'}.\n")


def evaluate_single_prediction(original_review, model, vectorizer):
    # Strip any extra HTML tags or newlines from the original review
    original_review_cleaned = original_review.replace('<br />', ' ').replace('\n', ' ').strip()

    # Display the cleaned original review
    print(f"Review (Original, unprocessed): {original_review_cleaned}")

    # Process the original review (apply all preprocessing steps)
    processed_review = preprocess_steps(original_review_cleaned)
    print(f"Processed Review: {processed_review}")

    # Vectorize the processed review for prediction
    review_vec = vectorizer.transform([processed_review])
    sentiment = model.predict(review_vec)[0]
    sentiment_prob = model.predict_proba(review_vec)

    print(f"Predicted Sentiment: {'positive' if sentiment == 'positive' else 'negative'}")
    print(explain_sentiment_probabilities(sentiment_prob))
    return sentiment



if __name__ == "__main__":
    try:
        model_filename = "latest_model.pkl"
        vectorizer_filename = "latest_vectorizer.pkl"

        model = joblib.load(model_filename)
        vectorizer = joblib.load(vectorizer_filename)

        print(f"Loaded the latest trained model: {model_filename}")
        print(f"Loaded the latest vectorizer: {vectorizer_filename}")
    except FileNotFoundError:
        print("Error: No previously trained model or vectorizer found.")
        sys.exit(1)

    # Load the test data from the most recently used dataset
    try:
        X_test, y_test, original_reviews = joblib.load("latest_test_data.pkl")  # Now also load original reviews
    except FileNotFoundError:
        print("Error: No test data found. Ensure you've trained a model before running this script.")
        sys.exit(1)

    # Randomly select a review for prediction
    random_index = random.randint(0, len(original_reviews) - 1)
    original_review = original_reviews.iloc[random_index]  # Original review from dataset

    # Evaluate the review (original and processed)
    evaluate_single_prediction(original_review, model, vectorizer)

    # Vectorize the test set before making predictions
    X_test_vec = vectorizer.transform(X_test)

    # Evaluate the model on the entire test set
    y_pred = model.predict(X_test_vec)
    y_pred_proba = model.predict_proba(X_test_vec)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    print(f"Evaluation on Test Set:")
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}")
    print("Confusion Matrix:")
    print(matrix)

    # Convert string labels to 0 and 1
    label_encoder = LabelEncoder()
    y_test_encoded = label_encoder.fit_transform(y_test)

    # Plot Confusion Matrix and ROC-AUC Curve
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.show()

    # Plot ROC-AUC Curve
    fpr, tpr, _ = roc_curve(y_test_encoded, y_pred_proba[:, 1])
    roc_auc = roc_auc_score(y_test_encoded, y_pred_proba[:, 1])

    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()
