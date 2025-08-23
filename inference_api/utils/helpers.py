from mlflow.exceptions import MlflowException, RestException
from mlflow.tracking import MlflowClient

def _get_by_alias(client: MlflowClient, name: str, alias: str):
    try:
        return client.get_model_version_by_alias(name=name, alias=alias)
    except (MlflowException, RestException):
        return None

def _latest_version(client: MlflowClient, name: str):
    vers = client.search_model_versions(f"name = '{name}'")
    if not vers:
        return None
    vers.sort(key=lambda v: int(v.last_updated_timestamp or 0), reverse=True)
    return vers[0]