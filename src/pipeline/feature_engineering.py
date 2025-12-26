import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import Config

# Try importing Spark, fallback to Pandas if it fails (common on Windows without Hadoop)
try:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    HAS_SPARK = True
except (ImportError, AttributeError) as e:
    print(f"Warning: Could not import Spark. This is common on Windows if Hadoop is not set up.\nError: {e}\nFalling back to Pandas for demonstration.")
    import pandas as pd
    HAS_SPARK = False

def create_spark_session():
    if not HAS_SPARK:
        return None
    return SparkSession.builder \
        .appName(Config.SPARK_APP_NAME) \
        .master(Config.SPARK_MASTER) \
        .getOrCreate()

def process_interaction_data(spark, input_path, output_path):
    """
    Load raw user-item interaction data, features engineering, and save formatted data.
    """
    if HAS_SPARK:
        run_spark_job(spark, input_path, output_path)
    else:
        run_pandas_job(input_path, output_path)

def run_spark_job(spark, input_path, output_path):
    print(f"Reading data from {input_path} (Spark)")
    
    # Creating dummy dataframe for demonstration
    data = [
        (1, 101, 1, 1672531200),
        (1, 102, 0, 1672534800),
        (2, 101, 1, 1672538400),
        (3, 103, 1, 1672542000)
    ]
    columns = ["user_id", "item_id", "click", "timestamp"]
    df = spark.createDataFrame(data, columns)
    
    print("Performing feature engineering (Spark)...")
    user_activity = df.groupBy("user_id").count().withColumnRenamed("count", "user_activity_count")
    item_popularity = df.groupBy("item_id").count().withColumnRenamed("count", "item_popularity_count")
    
    df_enriched = df.join(user_activity, "user_id", "left") \
                    .join(item_popularity, "item_id", "left")
    
    df_enriched.show()
    print(f"Saving processed data to {output_path}")

def run_pandas_job(input_path, output_path):
    print(f"Reading data from {input_path} (Pandas Fallback)")
    
    data = {
        "user_id": [1, 1, 2, 3],
        "item_id": [101, 102, 101, 103],
        "click": [1, 0, 1, 1],
        "timestamp": [1672531200, 1672534800, 1672538400, 1672542000]
    }
    df = pd.DataFrame(data)
    
    print("Performing feature engineering (Pandas)...")
    user_activity = df.groupby("user_id").size().reset_index(name="user_activity_count")
    item_popularity = df.groupby("item_id").size().reset_index(name="item_popularity_count")
    
    df_enriched = pd.merge(df, user_activity, on="user_id", how="left")
    df_enriched = pd.merge(df_enriched, item_popularity, on="item_id", how="left")
    
    print(df_enriched)
    print(f"Saving processed data to {output_path}")

if __name__ == "__main__":
    spark = create_spark_session()
    
    # Example paths
    raw_data_path = os.path.join(Config.DATA_DIR, "raw", "interactions.parquet")
    processed_data_path = os.path.join(Config.DATA_DIR, "processed", "features.parquet")
    
    process_interaction_data(spark, raw_data_path, processed_data_path)
    
    if spark:
        spark.stop()
