from data_preprocessing import preprocess_data
from sklearn.model_selection import train_test_split

def prepare_small_dataset():
    # Import data from `load_data_big`
    from load_data_big import prepare_data

    # Ensure data is loaded only once
    data = prepare_data()

    if data is not None:
        print(f"Original data size: {len(data)}")

        # Create a smaller subset of the data (5000 samples)
        data_subset = data.sample(n=5000, random_state=42)
        print(f"Subset data size: {len(data_subset)}")

        # Split into Training (80%), Validation (10%), and Test (10%)
        X_train, X_temp, y_train, y_temp = train_test_split(
            data_subset['processed_review'], data_subset['label'], test_size=0.2, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )

        # Split original reviews into Train, Validation, and Test sets
        original_train, original_temp = train_test_split(
            data_subset['original_review'], test_size=0.2, random_state=42
        )
        original_val, original_test = train_test_split(
            original_temp, test_size=0.5, random_state=42
        )

        # Output the size of the training, validation, and test sets
        print(f"Training samples (small dataset): {len(X_train)}")
        print(f"Validation samples (small dataset): {len(X_val)}")
        print(f"Test samples (small dataset): {len(X_test)}")

        return X_train, X_val, X_test, y_train, y_val, y_test, original_train, original_val, original_test  # Return original reviews as well
    else:
        print("Data loading failed.")
        return None, None, None, None, None, None, None, None, None

# Main block to execute the script
if __name__ == "__main__":
    print("Preparing small dataset...")
    prepare_small_dataset()
