
import numpy as np
import pandas as pd
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging

# logging configuration
logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('preprocessing_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Download required NLTK data
nltk.download('wordnet')
nltk.download('stopwords')

# Define the preprocessing function


def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:

        if comment is None or str(comment).strip() == "" or str(comment).lower() == "nan":
            return None   # or "" if you prefer keeping an empty string

        # Convert to lowercase
        comment = comment.lower()

        # Remove trailing and leading whitespaces
        comment = comment.strip()

        # Remove newline characters
        comment = re.sub(r'\n', ' ', comment)

        # Remove non-alphanumeric characters, except punctuation
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # Remove stopwords but retain important ones for sentiment analysis
        stop_words = set(stopwords.words('english')) - \
            {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join(
            [word for word in comment.split() if word not in stop_words])

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word)
                           for word in comment.split()])

        return comment
    except Exception as e:
        logger.error(f"Error in preprocessing comment: {e}")
        return comment


def normalize_text(df):
    """Apply preprocessing to the text data in the dataframe."""
    try:
        df['commentText'] = df['commentText'].apply(preprocess_comment)
        df['videoTitle'] = df['videoTitle'].apply(preprocess_comment)
        df['videoDescription'] = df['videoDescription'].apply(
            preprocess_comment)
        logger.debug('Text normalization completed')
        return df
    except Exception as e:
        logger.error(f"Error during text normalization: {e}")
        raise


def save_data(data: pd.DataFrame, data_path: str) -> None:
    """Save the processed train and test datasets."""
    try:
        interim_data_path = os.path.join(data_path, 'interim')
        logger.debug(f"Creating directory {interim_data_path}")

        # Ensure the directory is created
        os.makedirs(interim_data_path, exist_ok=True)
        logger.debug(
            f"Directory {interim_data_path} created or already exists")

        data.to_csv(os.path.join(interim_data_path,
                                 "data_processed.csv"), index=False)

        logger.debug(f"Processed data saved to {interim_data_path}")
    except Exception as e:
        logger.error(f"Error occurred while saving data: {e}")
        raise


def main():
    try:
        logger.debug("Starting data preprocessing...")

        # Fetch the data from data/raw
        data = pd.read_csv('./data/newraw/data.csv')
        logger.debug('Data loaded successfully')

        # Preprocess the data
        processed_data = normalize_text(data)

        # Save the processed data
        save_data(data,
                  data_path='./data')
    except Exception as e:
        logger.error(
            'Failed to complete the data preprocessing process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
