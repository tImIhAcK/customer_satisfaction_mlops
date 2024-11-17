import logging
from zenml import step
import pandas as pd
from typing import Optional

class IngestData:
    def __init__(self, data_path: str):
        self.data_path = data_path
        
    def get_data(self) -> pd.DataFrame:
        logging.info(f"Ingesting data from {str(self.data_path)}")
        return pd.read_csv(self.data_path)
     
        
@step
def ingest_df(data_path: str) -> Optional[pd.DataFrame]:
    """Ingesting data from the data path

    Args:
        data_path (str): path to data

    Returns:
        pd.DataFrame: the ingested dataframe
    """
    try:
        ingest_data = IngestData(data_path)
        return ingest_data.get_data()
    except Exception as e:
        logging.error(f"Error ingesting data... {str(e)}")
        raise ValueError(str(e))