from pipelines.training_pipeline import training_pipeline
from zenml.client import Client

if __name__ == "__main__":
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    training_pipeline("data/olist_customer_dataset.csv")
    
    

# zenml stack describe/list
# zenml logout
# zenml login --local
# zenml integration install mlflow -y
# zenml experiment-tracker register mlflow_tracker --flavor=mlflow
# zenml model-deployer register mlflow --flavor=mlflow
# zenml stack register mlflow_stack -a default -o default -d mlflow -e mlflow_tracker --set
# mlflow ui --backend-store-uri "file:/home/timihack/.config/zenml/local_stores/0b522111-6a70-40f1-b84a-aa45de851a80/mlruns"

