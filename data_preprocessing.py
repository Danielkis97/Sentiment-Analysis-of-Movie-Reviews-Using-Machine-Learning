import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import sys
import pandas as pd

# Ensure NLTK resources are downloaded
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize stop words and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Define a list of important words to keep (both sentiment words and important stopwords)
important_words = {
    "terrible", "bad", "awful", "horrible", "worst", "disgusting", "hate", "poor", "disappointing", "sucks",
    "boring", "pathetic", "negative", "failure", "mediocre",
    "good", "great", "excellent", "amazing", "fantastic", "wonderful", "love", "best", "awesome", "brilliant",
    "perfect", "positive", "enjoy", "superb", "outstanding",
    "not", "very", "really"
}

def preprocess_data(text):
    """
    Preprocesses the text by converting it to lowercase, removing non-alphabetic characters,
    removing stopwords, and lemmatizing.
    """
    # Show the original review without any preprocessing
    return text.lower()

def preprocess_steps(text):
    # Lowercase the text
    text = text.lower()

    # Tokenize the text (split on spaces)
    tokens = text.split()

    # Remove non-alphabetic tokens
    tokens = [word for word in tokens if word.isalpha()]

    # Remove stop words, but keep important words (both sentiment words and whitelisted stopwords)
    tokens = [word for word in tokens if word not in stop_words or word in important_words]

    # Lemmatize tokens
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Join tokens back to string
    processed_text = ' '.join(tokens)

    return processed_text

def prepare_dataset(dataset_choice):
    """
    Prepares the dataset based on the chosen dataset size (small or large).
    """
    if dataset_choice == "small":
        from load_data_small import prepare_small_dataset as prepare_data
    elif dataset_choice == "large":
        from load_data_big import prepare_data
    else:
        print("Invalid choice. Please choose 'small' or 'large'.")
        sys.exit(1)

    # Load the dataset
    data = prepare_data()

    if isinstance(data, pd.DataFrame):
        print(f"Data type of loaded data: {type(data)}")
        print(f"Data columns: {data.columns.tolist()}")
        print(f"Number of rows in data: {len(data)}")

        if 'review' in data.columns and 'label' in data.columns:
            # Show the original review
            data['original_review'] = data['review'].apply(preprocess_data)

            # Process the review by applying all preprocessing steps
            data['processed_review'] = data['review'].apply(preprocess_steps)

            # Split dataset (80% train, 20% validation + test)
            X_train, X_temp, y_train, y_temp = train_test_split(
                data['processed_review'], data['label'], test_size=0.2, shuffle=True
            )

            # Split the remaining 20% into validation and test
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=True)

            print(f"X_train size: {len(X_train)}")
            print(f"X_val size: {len(X_val)}")
            print(f"X_test size: {len(X_test)}")

            return X_train, X_val, X_test, y_train, y_val, y_test
        else:
            print("Data does not contain the required columns.")
            return None, None, None, None, None, None
    else:
        print("Data loading failed or returned an unexpected type.")
        return None, None, None, None, None, None
