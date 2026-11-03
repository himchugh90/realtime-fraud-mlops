"""Model registry conventions: one registered model, promoted by alias."""
import os

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

MODEL_NAME = os.getenv("MODEL_NAME", "fraud-detector")

CHAMPION = "champion"
CHALLENGER = "challenger"
PREVIOUS_CHAMPION = "previous_champion"


def version_for_alias(client: MlflowClient, alias: str):
    """Return the ModelVersion behind an alias, or None if the alias is unset."""
    try:
        return client.get_model_version_by_alias(MODEL_NAME, alias)
    except MlflowException:
        return None
