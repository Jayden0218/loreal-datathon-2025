import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml
import logging
import ast
import re


# Show all columns
pd.set_option("display.max_columns", None)

# Logging configuration
logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise


def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(data_url)
        logger.debug('Data loaded from %s', data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


def comment_preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data by handling missing values, duplicates, and empty strings."""
    try:
        # Removing missing values
        df.dropna(subset=["textOriginal"], inplace=True)

        df.drop(columns=["kind"], inplace=True)

        df.drop(columns=["updatedAt"], inplace=True)

        df.drop(columns=["parentCommentId"], inplace=True)

        df.drop(columns=["authorId"], inplace=True)

        df.drop(columns=["publishedAt"], inplace=True)

        df.drop(columns=["channelId"], inplace=True)

        df.drop(columns=["commentId"], inplace=True)

        logger.debug(
            'Data preprocessing completed: Missing values, duplicates, and empty strings removed.')
        return df
    except KeyError as e:
        logger.error('Missing column in the dataframe: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error during preprocessing: %s', e)
        raise


def video_preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data by handling missing values, duplicates, and empty strings."""
    try:
        df.drop(columns=["kind"], inplace=True)

        df.drop(columns=["commentCount"], inplace=True)

        df.drop(columns=["contentDuration"], inplace=True)

        df.drop(columns=["defaultLanguage"], inplace=True)

        df.drop(columns=["defaultAudioLanguage"], inplace=True)

        df.drop(columns=["channelId"], inplace=True)

        df["topicCategories"] = df["topicCategories"].apply(
            lambda x: ast.literal_eval(x) if isinstance(
                x, str) and x.startswith("[") else x
        )

        def shorten_link(link):
            if isinstance(link, str) and "wiki/" in link:
                return link.split("wiki/")[-1]
            return link

        df["topicCategories"] = df["topicCategories"].apply(
            lambda links: [shorten_link(l).replace("_", " ")
                           for l in links] if isinstance(links, list) else links
        )

        def extract_hashtags(title):
            if isinstance(title, str):
                # captures hashtags without the "#"
                return re.findall(r"#(\w+)", title)
            return []

        def clean_title(title):
            if isinstance(title, str):
                return re.sub(r"#\w+", "", title).strip()
            return title

        # Create new column for tags
        df["tag_from_title"] = df["title"].apply(extract_hashtags)

        # Clean up the original title
        df["title"] = df["title"].apply(clean_title)

        df["tags"] = df["tags"].apply(
            lambda x: ast.literal_eval(x) if isinstance(
                x, str) and x.startswith("[") else x
        )

        df["tags"] = df["tags"].apply(
            lambda x: x if isinstance(x, list) else [])

        df["tag_from_title"] = df["tag_from_title"].apply(
            lambda x: x if isinstance(x, list) else [])

        df["all_tags"] = df.apply(lambda row: list(
            set(row["tags"] + row["tag_from_title"])), axis=1)

        df.drop(columns=["tag_from_title"], inplace=True)
        df.drop(columns=["tags"], inplace=True)
        df.drop(columns=["favouriteCount"], inplace=True)

        logger.debug(
            'Data preprocessing completed: Missing values, duplicates, and empty strings removed.')
        return df
    except KeyError as e:
        logger.error('Missing column in the dataframe: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error during preprocessing: %s', e)
        raise


def save_data(data_needed: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets, creating the raw folder if it doesn't exist."""
    try:
        raw_data_path = os.path.join(data_path, 'newraw')

        # Create the data/raw directory if it does not exist
        os.makedirs(raw_data_path, exist_ok=True)

        # Save the train and test data
        data_needed.to_csv(os.path.join(
            raw_data_path, "data.csv"), index=False)
        # test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)

        logger.debug('Train and test data saved to %s', raw_data_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise


def main():
    try:
        # Load parameters from the params.yaml in the root directory
        params = load_params(params_path=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '../../params.yaml'))
        test_size = params['data_ingestion']['test_size']

        # Load data from the specified URL
        df_comments = load_data("./database/comments1.csv")
        df_videos = load_data("./database/videos.csv")

        # Preprocess the data for comments
        final_df_comments = comment_preprocess_data(df_comments)

        # Preprocess the data for videos
        final_df_videos = video_preprocess_data(df_videos)

        merged = df_comments.merge(
            df_videos,
            on="videoId",
            how="left"
        )

        merged = merged.rename(columns={
            "textOriginal": "commentText",
            "likeCount_x": "commentLikes",
            "publishedAt": "videoPublishedAt",
            "title": "videoTitle",
            "description": "videoDescription",
            "likeCount_y": "videoLikes",
        })

        # Save the split datasets and create the raw folder if it doesn't exist
        save_data(merged, data_path=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '../../data'))

    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
