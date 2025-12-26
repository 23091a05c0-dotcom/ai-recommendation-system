from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import sys
import os

# Define default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Python callable for validation
def validate_model_task():
    print("Validating model performance...")
    # Add validation logic here (e.g., compare with previous best model)
    return "Model Validation Passed"

# Define DAG
with DAG(
    'recsys_retraining',
    default_args=default_args,
    description='A DAG to retrain the recommendation model',
    schedule_interval=timedelta(days=1),
    catchup=False,
) as dag:

    # Task 1: Data Processing (Spark Job)
    # in production this might be SparkSubmitOperator
    data_processing = BashOperator(
        task_id='data_processing',
        bash_command=f'python "{os.path.join(os.environ.get("AIRFLOW_HOME", "."), "src", "pipeline", "feature_engineering.py")}"',
    )

    # Task 2: Model Training
    model_training = BashOperator(
        task_id='model_training',
        bash_command=f'python "{os.path.join(os.environ.get("AIRFLOW_HOME", "."), "src", "models", "train.py")}"',
    )

    # Task 3: Model Validation
    model_validation = PythonOperator(
        task_id='model_validation',
        python_callable=validate_model_task,
    )

    # Task Dependencies
    data_processing >> model_training >> model_validation
