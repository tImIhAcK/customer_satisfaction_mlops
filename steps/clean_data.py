import logging
import pandas as pd
from zenml import step
from typing_extensions import Annotated
from typing import Tuple

from src.data_cleaning import DataCleaning, DataDivideStrategy, DataPreprocessStrategy

@step
def clean_df(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"]
]:
    
    """
    Clean the data and divides into train and test sets

    Args:
        df (pd.DataFrame): input dataframe
    Returns:
        X_train: train set
        X_test: test label
        y_train: train set
        y_test: test label
    """
    try:
        preprocess_stratgey = DataPreprocessStrategy()
        data_cleaning = DataCleaning(df, preprocess_stratgey)
        preprocessed_data = data_cleaning.handle_data()
        
        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(preprocessed_data, divide_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        logging.info("Data cleaning finished")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error in cleaning data: {e}")
        raise e