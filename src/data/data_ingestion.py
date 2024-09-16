import numpy as np
import pandas as pd
import yaml
import os
import logging
from sklearn.model_selection import train_test_split # type: ignore

# Set up logging
logging.basicConfig(level=logging.INFO, filename='ml_pipeline.log', 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def load_params(params_path: str) -> float:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
            test_size = params['data_ingestion']['test_size']
            logging.info(f"Loaded test_size: {test_size} from {params_path}")
            return test_size
    except FileNotFoundError:
        logging.error(f"File {params_path} not found.")
        raise
    except KeyError:
        logging.error(f"Invalid structure in {params_path}. 'data_ingestion' or 'test_size' is missing.")
        raise
    except yaml.YAMLError as e:
        logging.error(f"YAML error in {params_path}: {str(e)}")
        raise

def read_data(url: str):
    try:
        df = pd.read_csv(url)
        logging.info(f"Data loaded successfully from {url}")
        return df
    except pd.errors.EmptyDataError:
        logging.error(f"No data found at {url}.")
        raise
    except pd.errors.ParserError:
        logging.error(f"Error parsing CSV data from {url}.")
        raise
    except Exception as e:
        logging.error(f"Unexpected error reading data from {url}: {str(e)}")
        raise

def process_data(df: pd.DataFrame):
    try:
        df.drop(columns=['tweet_id'], inplace=True)
        final_df = df[df['sentiment'].isin(['happiness', 'sadness'])]
        final_df['sentiment'].replace({'happiness': 1, 'sadness': 0}, inplace=True)
        logging.info("Data processing complete. Filtered for happiness and sadness sentiments.")
        return final_df
    except KeyError as e:
        logging.error(f"Column missing in DataFrame: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Error processing data: {str(e)}")
        raise

def save_data(data_path: str, train_data: pd.DataFrame, test_data: pd.DataFrame):
    try:
        os.makedirs(data_path, exist_ok=True)
        train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)
        logging.info(f"Data saved successfully to {data_path}")
    except Exception as e:
        logging.error(f"Error saving data to {data_path}: {str(e)}")
        raise

def main():
    logging.info("ML pipeline started.")
    
    try:
        # Load parameters
        test_size = load_params('params.yaml')

        # Read data
        df = read_data('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')

        # Process data
        final_df = process_data(df)

        # Split data
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        logging.info(f"Data split into train and test sets with test_size={test_size}")

        # Save data
        data_path = os.path.join("data", "raw")
        save_data(data_path, train_data, test_data)

    except Exception as e:
        logging.error(f"An error occurred during the ML pipeline: {str(e)}")
    
    logging.info("ML pipeline finished.")

if __name__ == "__main__":
    main()

