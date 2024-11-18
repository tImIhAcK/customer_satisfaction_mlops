import logging
import pandas as pd
from typing import Tuple
from typing_extensions import Annotated

import mlflow
from zenml import step
from zenml.client import Client

from src.evaluation import MSE, RMSE, R2
from sklearn.base import RegressorMixin

experiment_tracter = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracter.name)
def evaluate_model(
    model: RegressorMixin,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[
    Annotated[float, "r2_score"],
    Annotated[float, "rmse"]
]:
    """
    Evaluate the model using Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R2)
    """
    try:
        prediction = model.predict(X_test)
        mse_class = MSE()
        mse = mse_class.calculate_score(y_test, prediction)
        mlflow.log_metric("mse", mse)
        
        rmse_class = RMSE()
        rmse = rmse_class.calculate_score(y_test, prediction)
        mlflow.log_metric("rmse", rmse)
        print(f"RMSE : {rmse}")
        
        r2_class = R2()
        r2 = r2_class.calculate_score(y_test, prediction)
        mlflow.log_metric("r2", r2)
        print(f"R2 : {r2}")
        
        return r2, rmse
    except Exception as e:
        logging.error(f"Error in evaluating model: {e}")
        raise e
    