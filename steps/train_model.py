import logging
import pandas as pd

import mlflow
from zenml import step
from zenml.client import Client

from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> RegressorMixin:
    """Train model"""
    try:
        model = None
        config = ModelNameConfig()
        if config.model_name == "LinearRegression":
            mlflow.sklearn.autolog()
            model = LinearRegressionModel()
            train_model = model.train(X_train, y_train)
            return train_model
        else:
            raise ValueError(f"Model not supported: {config.model_name}")
    except Exception as e:
        logging.error(f"Error in training model: {e}")
        raise e