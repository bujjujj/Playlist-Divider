import pandas as pd
import os

def is_string_ascii(s):
    """
    A helper function that checks if a string contains only ASCII characters.
    Returns True if it's pure ASCII, False otherwise.
    This is a backwards-compatible way of doing .isascii().
    """
    try:
        s.encode('ascii')
        return True
    except UnicodeEncodeError:
        return False

def clean_training_data_csv():
    """
    Reads the existing training data, removes rows with non-ASCII characters
    in the artist or track columns, and saves it to a new, clean file.
    """
    input_filename = 'training_features.csv'
    output_filename = 'training_features_cleaned.csv'

    if not os.path.exists(input_filename):
        print(f"Error: The file '{input_filename}' was not found. Nothing to clean.")
        return

    print(f"Reading '{input_filename}'...")
    try:
        df = pd.read_csv(input_filename, encoding='latin-1')
    except Exception as e:
        print(f"Could not read the CSV file. Error: {e}")
        return

    original_rows = len(df)
    print(f"Loaded {original_rows} total rows.")

    # --- The Filtering Logic ---
    df['artist'] = df['artist'].fillna('').astype(str)
    df['track'] = df['track'].fillna('').astype(str)

    # UPDATED LINE: Replaced .str.isascii() with .apply(is_string_ascii) for compatibility
    mask = df['artist'].apply(is_string_ascii) & df['track'].apply(is_string_ascii)

    # Apply the mask to keep only the clean rows
    df_cleaned = df[mask]
    
    cleaned_rows = len(df_cleaned)
    rows_removed = original_rows - cleaned_rows
    print(f"Removed {rows_removed} rows with non-ASCII characters.")

    # --- Save the Cleaned Data ---
    if cleaned_rows > 0:
        print(f"Saving {cleaned_rows} clean rows to '{output_filename}'...")
        df_cleaned.to_csv(output_filename, index=False, encoding='utf-8-sig')
        print("Done!")
    else:
        print("No clean data was found to save.")


if __name__ == "__main__":
    clean_training_data_csv()