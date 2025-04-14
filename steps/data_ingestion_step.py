import pandas as pd
import os
from src.ingest_data import DataIngestorFactory
from zenml import step
from typing import NamedTuple


@step
def data_ingestion_step(file_path: str) -> pd.DataFrame:
    """Ingest data from a CSV file using the appropriate DataIngestor."""
    # Determine the file extension dynamically
    file_extension = os.path.splitext(file_path)[1]

    # Get the appropriate DataIngestor (now only supports .csv)
    data_ingestor = DataIngestorFactory.get_data_ingestor(file_extension)

    # Ingest the data and return it
    data = data_ingestor.ingest(file_path)
    return data