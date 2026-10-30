import sys
import mlflow
from mlflow import MlflowClient

nom = sys.argv[1]                          # fraude-automl ou fraude-manuel
mlflow.set_tracking_uri('http://127.0.0.1:5000')
client = MlflowClient()
versions = client.search_model_versions(f"name='{nom}'")
derniere = max(int(v.version) for v in versions)   # la plus recente
client.set_registered_model_alias(nom, 'production', str(derniere))
print(f'{nom} : version {derniere} promue (alias production)')
