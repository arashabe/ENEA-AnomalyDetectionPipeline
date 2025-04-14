from scripts.load_data import load_json_data, load_csv_data
from scripts.preprocessing import preprocess
from scripts.anomaly_detection import detect_anomalies
from scripts.ensemble import combine_ensemble_labels
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from datetime import datetime
from sqlalchemy import create_engine
import os



# Spark Initialization
spark = SparkSession.builder.appName("PODID Anomaly Detection").getOrCreate()

# Step 1 - JSON Data Loading
json_path = "data/Menowattge"
csv_path = "data/municipalities/municipalities_info.csv"
df_raw = load_json_data(spark, json_path)
municipalities_df = load_csv_data(spark, csv_path)

# Debug schema
#print("Schema :")
#df_raw.printSchema()

# Step 2 - Preprocessing with PySpark
df_processed = preprocess(df_raw, municipalities_df)
df_processed.printSchema()


# Step 3 - Anomaly Detection with PySpark
df_pandas = df_processed.toPandas().dropna()

df_with_labels = detect_anomalies(df_pandas)

# Step 4 - Ensemble
df_final = combine_ensemble_labels(df_with_labels)

# Step 5 - Results Output
print("Preview of detected anomalies:")
df_final_spark = spark.createDataFrame(df_final)
df_final_spark.filter(df_final_spark["anomaly_ensemble"] == 1).show()
df_final_spark_selected = df_final_spark.filter(df_final_spark["anomaly_ensemble"] == 1)
df_final_spark_selected = df_final_spark_selected.withColumn(
    "insertion_time",
    F.lit(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
)

# Step 6 - Prepare Data for Database Insertion
#output_dir = "output"
#df_final[df_final["anomaly_ensemble"] == 1].to_csv(f"{output_dir}/anomalies.csv", header=True, index=False)
# Convert PySpark DataFrame to Pandas DataFrame
df_selected = df_final_spark_selected.select(
    "PODID",
    "anomaly_pca_label",
    "anomaly_iso_label",
    "anomaly_lof_label",
    "anomaly_ensemble",
    "insertion_time"
).toPandas()



# Setting up MySQL Connection using Environment Variables
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DB = os.getenv("MYSQL_DB")
MYSQL_HOST = os.getenv("MYSQL_HOST")

if not all([MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB, MYSQL_HOST]):
    raise EnvironmentError("Make sure you have correctly set the environment variables MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB, and MYSQL_HOST.")

# Configure SQLAlchemy Engine
engine = create_engine(f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}/{MYSQL_DB}")

# Function to Insert Data in Bulk
def insert_data(df_selected):
    try:
        # Bulk insert using Pandas
        df_selected.to_sql('anomaly_results', con=engine, if_exists='append', index=False)
        print("The data has been successfully inserted!")
    except Exception as e:
        print(f"Error during insertion: {e}")

insert_data(df_selected)

