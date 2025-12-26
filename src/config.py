import os

class Config:
    # Paths
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
    
    # Model Hyperparameters
    EMBEDDING_DIM = 32
    HIDDEN_DIM = 64
    DROPOUT = 0.2
    LEARNING_RATE = 0.001
    BATCH_SIZE = 1024
    EPOCHS = 10

    # Spark Config
    SPARK_APP_NAME = "RecSysPipeline"
    SPARK_MASTER = "local[*]"
