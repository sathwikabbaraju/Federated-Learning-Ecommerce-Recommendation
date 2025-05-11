import os
import pandas as pd

# Define the required columns
REQUIRED_COLUMNS = ["name", "main_category", "sub_category", "ratings", "no_of_ratings", "discount_price", "actual_price", "purchased"]

# Path to synthetic dataset folder
DATASET_FOLDER = "synthetic_dataset"

def validate_csv(file_path):
    """Validate a single CSV file."""
    try:
        df = pd.read_csv(file_path)

        # Check if all required columns exist
        missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_columns:
            print(f"❌ ERROR: {file_path} is missing columns: {missing_columns}")
            return False

        # Check for null values
        if df.isnull().values.any():
            print(f"❌ ERROR: {file_path} contains missing values!")
            return False

        print(f"✅ {file_path} passed validation.")
        return True

    except Exception as e:
        print(f"❌ ERROR: Failed to process {file_path}: {str(e)}")
        return False


def main():
    all_files_valid = True

    for filename in os.listdir(DATASET_FOLDER):
        if filename.endswith(".csv"):
            file_path = os.path.join(DATASET_FOLDER, filename)
            if not validate_csv(file_path):
                all_files_valid = False

    if not all_files_valid:
        print("❌ Validation failed. Please fix errors before merging!")
        exit(1)  # Exit with error to fail the GitHub Action
    else:
        print("✅ All CSV files passed validation!")

if __name__ == "__main__":
    main()