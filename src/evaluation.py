import logging
import numpy as np
from abc import ABC, abstractmethod

from sklearn.metrics import mean_squared_error, r2_score
class Evaluation(ABC):
    """Abstract class defining evaluation strategy for evaluating our model"""
    @abstractmethod
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Calculates the score for the model
        Args:
            y_true: True label
            y_pred: Predicted label
        Returns:
            None
        """
        pass

class MSE(Evaluation):
    """Mean Squared Error evaluation strategy"""
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating MSE")
            mse = mean_squared_error(y_true, y_pred)
            return mse
        except Exception as e:
            logging.error(f"Error in calculating MSE: {e}")
            raise e
        
class R2(Evaluation):
    """R-squared evaluation strategy"""
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating R2")
            r2 = r2_score(y_true, y_pred)
            return r2
        except Exception as e:
            logging.error(f"Error in calculating R2: {e}")
            raise e
        
class RMSE(Evaluation):
    """Root Mean Squared Error evaluation strategy"""
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating RMSE")
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            return rmse
        except Exception as e:
            logging.error(f"Error in calculating RMSE: {e}")
            raise e