# PODID Anomaly Detection Pipeline

## Overview
The **PODID Anomaly Detection Pipeline** is a data processing and machine learning framework designed to identify anomalies in energy consumption data. Using PySpark for scalable data preprocessing and a combination of machine learning techniques, the pipeline optimizes unsupervised anomaly detection methods to generate reliable ensemble predictions.

This project leverages tools like PCA, Isolation Forest, and Local Outlier Factor to flag anomalous patterns, ensuring robust insights into energy data analysis.

---

## Features
- **Scalable Preprocessing**: Handles large-scale datasets efficiently using PySpark.
- **Unsupervised Anomaly Detection**:
  - Principal Component Analysis (PCA).
  - Isolation Forest with parameter optimization.
  - Local Outlier Factor (LOF) with parameter optimization.
- **Ensemble Method**: Combines predictions from multiple anomaly detection models for increased accuracy.
- **Customizable Output**: Exports detected anomalies in CSV format for further analysis.
- **Technology Integration**: Combines Python, PySpark, Apache Hadoop, scikit-learn, and more for a seamless workflow.

---

## Dataset Description
The dataset consists of energy consumption data and associated features. Below is a detailed description of each column:

| **Column Name**        | **Type**  | **Description**                                                                 |
|-------------------------|-----------|---------------------------------------------------------------------------------|
| `latitude`             | Double    | Latitude of the measurement location.                                           |
| `longitude`            | Double    | Longitude of the measurement location.                                          |
| `height`               | Double    | Height (in meters) of the measurement device.                                   |
| `start_ts`             | Timestamp | Start timestamp of the measurement period.                                      |
| `end_ts`               | Timestamp | End timestamp of the measurement period.                                        |
| `ElectricPanelID`      | String    | Unique identifier for the electric panel.                                       |
| `Line1Current`         | Double    | Electric current on Line 1 (in amperes).                                        |
| `Line2Current`         | Double    | Electric current on Line 2 (in amperes).                                        |
| `Line3Current`         | Double    | Electric current on Line 3 (in amperes).                                        |
| `PODID`                | String    | Unique identifier for the Point of Delivery (POD).                              |
| `Phase1ActivePower`    | Double    | Active power on Phase 1 (in kilowatts).                                         |
| `Phase1ApparentPower`  | Double    | Apparent power on Phase 1 (in kilovolt-amperes).                                |
| `Phase1PowerFactor`    | Double    | Power factor on Phase 1 (ratio of real power to apparent power).                |
| `Phase1ReactivePower`  | Double    | Reactive power on Phase 1 (in kilovars).                                        |
| `Phase1Voltage`        | Double    | Voltage on Phase 1 (in volts).                                                  |
| `Phase2ActivePower`    | Double    | Active power on Phase 2 (in kilowatts).                                         |
| `Phase2ApparentPower`  | Double    | Apparent power on Phase 2 (in kilovolt-amperes).                                |
| `Phase2PowerFactor`    | Double    | Power factor on Phase 2 (ratio of real power to apparent power).                |
| `Phase2ReactivePower`  | Double    | Reactive power on Phase 2 (in kilovars).                                        |
| `Phase2Voltage`        | Double    | Voltage on Phase 2 (in volts).                                                  |
| `Phase3ActivePower`    | Double    | Active power on Phase 3 (in kilowatts).                                         |
| `Phase3ApparentPower`  | Double    | Apparent power on Phase 3 (in kilovolt-amperes).                                |
| `Phase3PowerFactor`    | Double    | Power factor on Phase 3 (ratio of real power to apparent power).                |
| `Phase3ReactivePower`  | Double    | Reactive power on Phase 3 (in kilovars).                                        |
| `Phase3Voltage`        | Double    | Voltage on Phase 3 (in volts).                                                  |
| `TotalActiveEnergy`    | Double    | Total energy consumption during the period (in kilowatt-hours).                 |
| `TotalActivePower`     | Double    | Total active power (in kilowatts).                                              |
| `TotalApparentPower`   | Double    | Total apparent power (in kilovolt-amperes).                                     |
| `TotalReactiveEnergy`  | Double    | Total reactive energy during the period (in kilovars-hours).                    |
| `TotalReactivePower`   | Double    | Total reactive power (in kilovars).                                             |
| `TownCode`             | String    | Code representing the town or location of the measurement.                      |

---

## Technologies Used
This project employs the following technologies and tools:

1. **Python**: The primary programming language used for writing the project code.
2. **PySpark**: Used for distributed data processing and handling large datasets. PySpark integrates with Apache Spark, enabling scalable operations on data. `SparkSession` was used to initialize PySpark and interact with the data.
3. **Apache Hadoop**: Integrated to manage data in a distributed manner. Features like HDFS (Hadoop Distributed File System) and the native `libhadoop` library were configured to process data distributedly via Spark.
4. **Scikit-learn**: Used for machine learning and anomaly detection, implementing algorithms such as:
   - PCA (Principal Component Analysis): For dimensionality reduction and anomaly detection through reconstruction error.
   - Isolation Forest: For anomaly detection based on identifying data points that deviate from "normal" behavior.
   - Local Outlier Factor (LOF): For measuring data point isolation relative to its neighbors.
5. **Pandas**: Utilized for data manipulation and analysis. Data from PySpark was converted into Pandas DataFrames for easier handling during anomaly detection.
6. **NumPy**: Used for mathematical and numerical computations, essential for operations like variance calculations and array manipulations required for PCA and other machine learning algorithms.
7. **VS Code (IDE)**: Development was carried out using VS Code as the Integrated Development Environment, with its integrated terminal for executing PySpark commands and interacting with Hadoop and Spark.
8. **Bash (Linux/Ubuntu)**: The Ubuntu terminal and Bash commands were used to configure the environment, install necessary packages, and set up environment variables like `JAVA_LIBRARY_PATH` to ensure Hadoop and PySpark integration.
9. **Apache Spark**: The distributed computing platform that works with PySpark to process large datasets in parallel, enabling efficient data transformations and analysis.
10. **Silhouette Score**: A metric from scikit-learn used to evaluate the quality of clustering and anomaly detection models. It measures how well a data point fits within its assigned cluster compared to others, facilitating the detection of anomalies in the dataset.

---

## Installation

### Prerequisites
Ensure the following are installed:
- **Python**: Version 3.10.14 or later.
- **Apache Spark**: Version 3.3.0.
- **Hadoop**: Version 3.3.6.
- Python libraries:
  - PySpark (`pyspark`)
  - NumPy (`numpy`)
  - Pandas (`pandas`)
  - Scikit-learn (`sklearn`)

### Setup
1. Clone the repository:
   ```bash
   git clone https://sccserver.enea.it/arash/enea-anomalydetectionpipeline.git
   cd enea-anomalydetectionpipeline
