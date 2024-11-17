import logging
from abc import ABC, abstractmethod

from sklearn.linear_model import LinearRegression


class Model(ABC):
    """Abstract base class for models"""
    @abstractmethod
    def train(self, X_train, y_train):
        """Train the model
        Args:
            X_train: Training data
            y_train: Training labels
        """
        pass
    
class LinearRegressionModel(Model):
    """Linear regression model"""
    def train(self, X_train, y_train, **kwargs):
       """Train the model
       Args:
            X_train: Training data
            y_train: Training labels
        Returns:
         None
       """
       try:   
           reg = LinearRegression(**kwargs)
           reg.fit(X_train, y_train)
           logging.info("Model trained successfully")
           return reg
       except Exception as e:
           logging.error(f"Error in training model: {e}")
           raise e