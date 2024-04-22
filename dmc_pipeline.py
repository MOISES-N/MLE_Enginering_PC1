import airflow
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from joblib import dump, load
from GetDataKaggle import GetDataKaggle
from automl import AutoML_PyCaret, SubmitKaggle
from test import TestModelo

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 4, 20),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'ml_workflow_demo',
    default_args=default_args,
    description='A simple ML pipeline',
    schedule_interval='0 17 * * *',
)

descarga_data = PythonOperator(
    task_id='GetDataKaggle',
    python_callable=GetDataKaggle,
    dag=dag,
)

auto_ml = PythonOperator(
    task_id='AutoML_PyCaret',
    python_callable=AutoML_PyCaret,
    dag=dag,
)

evaluate_model_task = PythonOperator(
    task_id='TestModelo',
    python_callable=TestModelo,
    dag=dag,
)

submit_results = PythonOperator(
    task_id='SubmitKaggle',
    python_callable=SubmitKaggle,
    dag=dag,
)



descarga_data >> auto_ml >> evaluate_model_task >> submit_results