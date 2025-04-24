from pyspark.sql.functions import col, isnull, count, avg, sum as _sum, when
from pyspark.sql import DataFrame


def preprocess(df: DataFrame, municipalities_df: DataFrame) -> DataFrame:
    print("Preprocessing started ...")

    # Step 1: Join with the municipalities dataset
    df = df.join(municipalities_df, df["TownCode"] == municipalities_df["istat"], how="inner")

    # Step 2: Normalize numerical columns by population and surface area
    numeric_columns = [
        'Phase1Voltage', 'Line1Current', 'Phase1PowerFactor', 'Phase1ActivePower',
        'Phase1ApparentPower', 'Phase1ReactivePower', 'Phase2Voltage', 'Line2Current',
        'Phase2PowerFactor', 'Phase2ActivePower', 'Phase2ApparentPower',
        'Phase2ReactivePower', 'Phase3Voltage', 'Line3Current', 'Phase3PowerFactor',
        'Phase3ActivePower', 'Phase3ApparentPower', 'Phase3ReactivePower',
        'TotalActiveEnergy', 'TotalReactiveEnergy', 'TotalActivePower',
        'TotalApparentPower', 'TotalReactivePower'
    ]

    for colname in numeric_columns:
        if colname in df.columns:
            df = df.withColumn(
                f"{colname}_normalized",
                (col(colname) / col("population")).cast("double")
            )

    # Step 3: Group by PODID and month, and calculate sum and mean of TotalActiveEnergy
    df_grouped = df.groupBy("PODID", "month").agg(
        _sum("TotalActiveEnergy_normalized").alias("TotalActiveEnergy_sum"),
        avg("TotalActiveEnergy_normalized").alias("TotalActiveEnergy_mean")
    )

    # Step 4: Show the total number of records
    print("\nStep 2 - Total records:", df.count())



    # Step 5: Show the number of distinct PODIDs
    print("\nStep 3 - Distinct PODIDs:", df.select("PODID").distinct().count())

    # Step 6: Identify all distinct months in the dataset
    months_in_data = sorted([row["month"] for row in df.select("month").distinct().collect()])
    print("\nDistinct months in dataset:", months_in_data)

    total_months = len(months_in_data)

    podid_month_counts = df_grouped.groupBy("PODID").count()
    podids_with_all_months = podid_month_counts.filter(col("count") == total_months).select("PODID")

    df_filtered = df_grouped.join(podids_with_all_months, on="PODID", how="inner")

    # Step 7: Pivot the data so that each PODID becomes a row, with each month as a column
    df_pivoted_sum = df_filtered.groupBy("PODID").pivot("month").agg(
        _sum("TotalActiveEnergy_sum")
    )

    df_pivoted_mean = df_filtered.groupBy("PODID").pivot("month").agg(
        avg("TotalActiveEnergy_mean")
    )

    # Rename columns for better readability
    for col_name in df_pivoted_sum.columns:
        if col_name != "PODID":
            df_pivoted_sum = df_pivoted_sum.withColumnRenamed(col_name, f"TotalActiveEnergy_sum_{col_name}")
    for col_name in df_pivoted_mean.columns:
        if col_name != "PODID":
            df_pivoted_mean = df_pivoted_mean.withColumnRenamed(col_name, f"TotalActiveEnergy_mean_{col_name}")

    # Join the pivoted dataframes
    df_final = df_pivoted_sum.join(df_pivoted_mean, on="PODID")

    # Final statistics
    print("\nFinal pivoted DataFrame:")
    print("Total records:", df_final.count())
    print("Distinct PODIDs:", df_final.select("PODID").distinct().count())
    print("\nStep 4 - Nulls per column:")
    df_final.select([count(when(isnull(c), c)).alias(c) for c in df_final.columns]).show()

    print("Preprocessing ended successfully.")
    return df_final
