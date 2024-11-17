import logging
from abc import ABC, abstractmethod
from typing import Union

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
    """Abstract class defining strategy for habndling data"""
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]: # Return dataframe or series
        pass
    
class DataPreprocessStrategy(DataStrategy):
    """Strategy for preprocessing data"""
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data"""
        try:
            data = data.drop(
                [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp"
                ],
                axis=1
            )
            data["product_weight_g"] = data["product_weight_g"].fillna(data["product_weight_g"].median())
            data["product_length_cm"] = data["product_length_cm"].fillna(data["product_length_cm"].median())
            data["product_height_cm"] = data["product_height_cm"].fillna(data["product_height_cm"].median())
            data["product_width_cm"] = data["product_width_cm"].fillna(data["product_width_cm"].median())
            data["review_comment_message"] = data["review_comment_message"].fillna("No review")
            
            data = data.select_dtypes(include=[np.number])
            for col in data.columns:
                data[col] = data[col].fillna(data[col].median())
            cols_to_drop = ["customer_zip_code_prefix", "order_item_id"]
            data = data.drop(cols_to_drop, axis=1)
            return data
        except Exception as e:
            logging.error(f"Error in preprocessing data {e}")
            raise e
        
class DataDivideStrategy(DataStrategy):
    """Strategy for splitting into train ans test"""
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """Split into train and test"""
        try: 
            X = data.drop(['review_score'], axis=1)
            y = data['review_score']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error in splitting data: {e}")
            raise e
        
        
class DataCleaning:
    """Class for preprocessing and splitting the data"""
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy
        
    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error(f"Error in handling data: {e}")
            raise e
    