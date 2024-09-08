import os
import pandas as pd
from data_preprocessing import preprocess_data
from sklearn.model_selection import train_test_split

def load_reviews(directory):
    reviews = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                reviews.append(file.read())
    return reviews

def prepare_data():
    # Load the positive and negative reviews
    pos_reviews = load_reviews('D:/PycharmProjects/NLP/data/aclImdb/train/pos')
    neg_reviews = load_reviews('D:/PycharmProjects/NLP/data/aclImdb/train/neg')

    if not pos_reviews or not neg_reviews:
        print("No data loaded. Please check the paths and files.")
        return None

    # Combine reviews into a DataFrame and save both the original and processed versions
    data = pd.DataFrame({
        'review': pos_reviews + neg_reviews,
        'label': ['positive'] * len(pos_reviews) + ['negative'] * len(neg_reviews)
    })

    # Save original review before preprocessing
    data['original_review'] = data['review']
    # Preprocess the review
    data['processed_review'] = data['review'].apply(preprocess_data)

    print(f"Loaded positive reviews: {len(pos_reviews)}")
    print(f"Loaded negative reviews: {len(neg_reviews)}")

    return data

def prepare_dataset():
    data = prepare_data()  # Load the dataset

    if data is not None:
        # Split dataset for processed reviews
        X_train, X_temp, y_train, y_temp = train_test_split(
            data['processed_review'], data['label'], test_size=0.2, shuffle=True, random_state=42
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, shuffle=True, random_state=42
        )

        # Split the original reviews into train, validation, and test sets
        original_train, original_temp = train_test_split(
            data['original_review'], test_size=0.2, shuffle=True, random_state=42
        )

        original_val, original_test = train_test_split(
            original_temp, test_size=0.5, shuffle=True, random_state=42
        )

        # Return processed reviews, labels, and original reviews
        return X_train, X_val, X_test, y_train, y_val, y_test, original_train, original_val, original_test

    else:
        print("Data loading failed.")
        return None, None, None, None, None, None, None, None, None

# Example usage
if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test, original_train, original_val, original_test = prepare_dataset()

    if X_train is not None:
        print(f"Training samples (large dataset): {len(X_train)}")
        print(f"Validation samples (large dataset): {len(X_val)}")
        print(f"Test samples (large dataset): {len(X_test)}")