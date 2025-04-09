# main.py

from scripts.load_data import load_json_data
from scripts.preprocessing import preprocess
from scripts.anomaly_detection import detect_anomalies
from scripts.ensemble import combine_ensemble_labels
from pyspark.sql import SparkSession
import os



# Spark Initialization
spark = SparkSession.builder.appName("PODID Anomaly Detection").getOrCreate()

# Step 1 - JSON Data Loading
json_path = "data/Menowattge"
df_raw = load_json_data(spark, json_path)

# Debug schema
#print("Schema :")
#df_raw.printSchema()

# Step 2 - Preprocessing with PySpark
df_processed = preprocess(df_raw)
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


# Saving to disk (optional)
#output_dir = "output"
#df_final[df_final["anomaly_ensemble"] == 1].to_csv(f"{output_dir}/anomalies.csv", header=True, index=False)

