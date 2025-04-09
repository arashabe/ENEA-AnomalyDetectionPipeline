import time
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.model_selection import ParameterGrid

# Function for unsupervised anomaly detection model optimization
def optimize_unsupervised_model(X_scaled, model_type="isolation_forest"):
    best_model = None
    best_score = -1
    best_params = {}

    if model_type == "isolation_forest":
        param_grid = {
            "n_estimators": [50, 100],
            "contamination": [0.01, 0.05],
            "max_features": [0.5, 1.0]
        }
        ModelClass = IsolationForest
    elif model_type == "lof":
        param_grid = {
            "n_neighbors": [10, 20, 30],
            "contamination": [0.01, 0.05]
        }
        ModelClass = LocalOutlierFactor
    else:
        raise ValueError("Unsupported model type")

    print(f"\n🔍 Starting grid search for {model_type}...")
    for params in ParameterGrid(param_grid):
        try:
            if model_type == "isolation_forest":
                model = ModelClass(**params, random_state=42)
                labels = model.fit_predict(X_scaled)
            else:
                model = ModelClass(**params)
                labels = model.fit_predict(X_scaled)

            score = silhouette_score(X_scaled, labels)
            print(f"Params: {params} → Silhouette Score: {score:.4f}")

            if score > best_score:
                best_score = score
                best_model = model
                best_params = params
        except Exception as e:
            print(f"⚠️ Skipped params {params} due to error: {e}")

    print(f"✅ Best score for {model_type}: {best_score:.4f} with params: {best_params}")
    return best_model, best_params

# Main anomaly detection function
def detect_anomalies(df_pandas):
    print("📋 All columns in df_pandas:", df_pandas.columns.tolist())
    print("📊 Data types:\n", df_pandas.dtypes)

    # Step 1: Remove non-feature columns
    features = df_pandas.drop(columns=["PODID"])

    # Step 2: Keep only numeric columns
    features = features.select_dtypes(include=[np.number])

    # Step 3: Drop columns with NaNs
    features = features.dropna(axis=1, how="any")

    print("✅ Features selected for anomaly detection:", features.columns.tolist())
    print("📐 Feature matrix shape:", features.shape)

    # Step 4: Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # 🧠 PCA
    start_time = time.time()
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)
    print("Explained variance ratio by component:", pca.explained_variance_ratio_)
    print("This is a list of values that indicate how much of the total variance in the data is explained by each individual principal component.")
    print("Total variance explained:", np.sum(pca.explained_variance_ratio_))
    print("Number of PCA components used:", pca.n_components_)
    X_reconstructed = pca.inverse_transform(X_pca)
    reconstruction_error = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)
    threshold = np.percentile(reconstruction_error, 95)
    df_pandas["anomaly_pca_label"] = (reconstruction_error > threshold).astype(int)
    elapsed_pca = time.time() - start_time
    print(f"🧩 PCA completed in {elapsed_pca:.2f} seconds.")
    print("📈 PCA threshold (95th percentile):", threshold)

    # 🌲 Isolation Forest with optimization
    start_time = time.time()
    best_iso_model, best_iso_params = optimize_unsupervised_model(X_scaled, model_type="isolation_forest")
    iso_labels = best_iso_model.fit_predict(X_scaled)
    df_pandas["anomaly_iso_label"] = (iso_labels == -1).astype(int)
    elapsed_iso = time.time() - start_time
    print(f"🌲 Optimized Isolation Forest completed in {elapsed_iso:.2f} seconds.")

    try:
        sil_score_iso = silhouette_score(X_scaled, iso_labels)
        print("The Silhouette Score measures how well a data point fits within its assigned cluster compared to other clusters, ranging from -1 (poor fit) to +1 (strong fit).")
        print(f"🧪 Isolation Forest Silhouette Score: {sil_score_iso:.4f}")
    except Exception as e:
        print("⚠️ Could not compute Silhouette Score for Isolation Forest:", e)

    # 📉 LOF with optimization
    start_time = time.time()
    best_lof_model, best_lof_params = optimize_unsupervised_model(X_scaled, model_type="lof")
    lof_labels = best_lof_model.fit_predict(X_scaled)
    df_pandas["anomaly_lof_label"] = (lof_labels == -1).astype(int)
    elapsed_lof = time.time() - start_time
    print(f"📉 Optimized LOF completed in {elapsed_lof:.2f} seconds.")

    try:
        sil_score_lof = silhouette_score(X_scaled, lof_labels)
        print(f"🧪 LOF Silhouette Score: {sil_score_lof:.4f}")
    except Exception as e:
        print("⚠️ Could not compute Silhouette Score for LOF:", e)

    print("✅ Anomaly detection finished.")
    return df_pandas
