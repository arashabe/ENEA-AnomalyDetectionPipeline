def combine_ensemble_labels(df):
    df["anomaly_ensemble"] = (
        (df["anomaly_pca_label"] + df["anomaly_iso_label"] + df["anomaly_lof_label"]) >= 2
    ).astype(int)
    return df
