from pyspark.sql.functions import col, isnull, count, avg, sum as _sum, when
from pyspark.sql import DataFrame


def preprocess(df: DataFrame) -> DataFrame:
    print("Preprocessing started ...")
    # Step 1: Select only the required columns
    df = df.select("PODID", "month", "TotalActiveEnergy")

    # Step 2: Show total number of records
    print("\n✅ Step 2 - Total records:", df.count())

    # Step 3: Show number of nulls per column
    print("\n📊 Step 3 - Nulls per column:")
    df.select([count(when(isnull(c), c)).alias(c) for c in df.columns]).show()

    # Step 4: Show number of distinct PODIDs
    print("\n🆔 Step 4 - Distinct PODIDs:", df.select("PODID").distinct().count())

    # Step 5: Group by PODID and month, and compute sum and mean of TotalActiveEnergy
    df_grouped = df.groupBy("PODID", "month").agg(
        _sum("TotalActiveEnergy").alias("TotalActiveEnergy_sum"),
        avg("TotalActiveEnergy").alias("TotalActiveEnergy_mean")
    )

    # Step 6: Show stats after aggregation
    #print("\n✅ After groupBy(PODID, month):")
    #print("Total records:", df_grouped.count())
    #print("Distinct PODIDs:", df_grouped.select("PODID").distinct().count())
    #print("Nulls per column:")
    #df_grouped.select([count(when(isnull(c), c)).alias(c) for c in df_grouped.columns]).show()

    # Step 7: Filter PODIDs that have records for ALL months
    months_in_data = sorted([row["month"] for row in df.select("month").distinct().collect()])
    print("\n🗓️ Distinct months in dataset:", months_in_data)
    total_months = len(months_in_data)

    podid_month_counts = df_grouped.groupBy("PODID").count()
    podids_with_all_months = podid_month_counts.filter(col("count") == total_months).select("PODID")

    df_filtered = df_grouped \
    .join(podids_with_all_months, on="PODID", how="inner") \
    .filter(col("month").isin(months_in_data)) 

    # Step 8: Stats after filtering for complete PODIDs
    #print("\n✅ After filtering PODIDs with all", total_months, "months:")
    #print("Total records:", df_filtered.count())
    #print("Distinct PODIDs:", df_filtered.select("PODID").distinct().count())
    #print("Nulls per column:")
    #df_filtered.select([count(when(isnull(c), c)).alias(c) for c in df_filtered.columns]).show()

    # Step 9: Pivot the DataFrame so each PODID is a row and each month becomes a column
    # Pivot sum and mean separately
    df_pivoted_sum = df_filtered.groupBy("PODID").pivot("month").agg(
        _sum("TotalActiveEnergy_sum")
    )

    df_pivoted_mean = df_filtered.groupBy("PODID").pivot("month").agg(
        avg("TotalActiveEnergy_mean")
    )

    # Rename columns to meaningful names
    for col_name in df_pivoted_sum.columns:
        if col_name != "PODID":
            df_pivoted_sum = df_pivoted_sum.withColumnRenamed(col_name, f"TotalActiveEnergy_sum_{col_name}")

    for col_name in df_pivoted_mean.columns:
        if col_name != "PODID":
            df_pivoted_mean = df_pivoted_mean.withColumnRenamed(col_name, f"TotalActiveEnergy_mean_{col_name}")

    # Join the pivoted tables
    df_final = df_pivoted_sum.join(df_pivoted_mean, on="PODID")

    # Final stats
    print("\n✅ Final pivoted DataFrame:")
    print("Total records:", df_final.count())
    print("Distinct PODIDs:", df_final.select("PODID").distinct().count())
    #print("Nulls per column:")
    #df_final.select([count(when(isnull(c), c)).alias(c) for c in df_final.columns]).show()
    print("✅ Preprocessing ended successfully.")

    return df_final



