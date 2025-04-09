from pyspark.sql.functions import (
    col, explode, expr, to_timestamp, month, when
)
from pyspark.sql.types import DoubleType


def load_json_data(spark, folder_path):
    print(f"📂 Loading JSON files from: {folder_path} using Spark")

    # Load all JSON lines from files in the folder (NDJSON format)
    df_raw = spark.read.json(folder_path)  # Automatically parallel, line-by-line

    # Filter only valid entries
    df_valid = df_raw.filter("UrbanDataset.values.line is not null")

    # Explode the list of lines in each object
    df_exploded = df_valid.selectExpr("explode(UrbanDataset.values.line) as line")

    # Extract main fields
    df_flat = df_exploded.select(
        col("line.coordinates.latitude").alias("latitude"),
        col("line.coordinates.longitude").alias("longitude"),
        col("line.coordinates.height").alias("height"),
        col("line.period.start_ts").alias("start_ts"),
        col("line.period.end_ts").alias("end_ts"),
        col("line.property").alias("properties")
    )

    # Transform property list into key-value structure
    df_kv = df_flat.select(
        "*",
        expr("transform(properties, x -> struct(x.name as key, x.val as value)) as kv_pairs")
    ).drop("properties")

    df_exploded_kv = df_kv.withColumn("kv", explode("kv_pairs")).drop("kv_pairs")

    df_with_id = df_exploded_kv.withColumn(
        "record_id", expr("concat_ws('_', start_ts, end_ts)")
    )

    df_pivoted = df_with_id.groupBy("record_id").pivot("kv.key").agg(expr("first(kv.value)"))

    df_static = df_with_id.select(
        "record_id", "latitude", "longitude", "height",
        to_timestamp("start_ts").alias("start_ts"),
        to_timestamp("end_ts").alias("end_ts")
    ).distinct()

    df_total = df_static.join(df_pivoted, on="record_id", how="inner").drop("record_id")

    df_total = df_total.withColumn("month", month("start_ts"))

    # Replace string "null" or empty string with real null
    for colname in df_total.columns:
        df_total = df_total.withColumn(
            colname,
            when((col(colname) == "") | (col(colname) == "null"), None).otherwise(col(colname))
        )

    numeric_columns = [
        'Phase1Voltage', 'Line1Current', 'Phase1PowerFactor', 'Phase1ActivePower',
        'Phase1ApparentPower', 'Phase1ReactivePower', 'Phase2Voltage', 'Line2Current',
        'Phase2PowerFactor', 'Phase2ActivePower', 'Phase2ApparentPower',
        'Phase2ReactivePower', 'Phase3Voltage', 'Line3Current', 'Phase3PowerFactor',
        'Phase3ActivePower', 'Phase3ApparentPower', 'Phase3ReactivePower',
        'TotalActiveEnergy', 'TotalReactiveEnergy', 'TotalActivePower',
        'TotalApparentPower', 'TotalReactivePower'
    ]

    for c in numeric_columns:
        if c in df_total.columns:
            df_total = df_total.withColumn(c, col(c).cast(DoubleType()))

    print("✅ Finished loading.")
    print(f"📊 Total rows: {df_total.count()}")
    #print("🆔 Distinct PODIDs:")
    #df_total.select("PODID").distinct().show()

    return df_total

