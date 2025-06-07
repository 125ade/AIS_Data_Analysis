<div align="center"> 
  <img src="ais_logo.webp" alt="Logo AIS Data Analysis" width="200" height="200"/>
</div>

<h1 align="center">AIS Data Analysis</h1>

## Quick Start Guide

### Step 1: Install Dependencies

To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
```

### Step 2: Create Required Directories

Create the following directories for your dataset:

```bash
mkdir -p "dataset/AIS_Dataset"
mkdir -p "dataset/AIS_Dataset_csv"
```

### Step 3: Add Dataset Files

Place the files named `ais_stat_data_{year}.csv` into the `dataset/AIS_Dataset` directory.

### Step 4: Prepare CSV Files

Run the following script to organize and prepare the CSV files:

```bash
python setupCSV.py
```

### Step 5: Prepare Tracks CSV Files

Run the following script to organize and prepare the vessels tracks CSV files, you need to select the year

```bash
python setupTracks.py --year 2020
```

### Step 6: Run Data Analysis

Initiate a preliminary analysis of the dataset using:

```bash
python analyzer_1.py
```

## Overview of Analysis

The project workflows are organized into several phases and types of analysis:

### 1. Data Preparation & Cleaning
- **setupCSV.py**: Imports raw AIS files and generates cleaned CSVs with standardized columns.
- **setupTracks.py**: Extracts and normalizes track trajectories from AIS records for plotting.
- **setupSplitUnder10.py**: Filters out tracks with fewer than 10 points.
- **setupClassification.py**: Builds a structured dataset ready for classification models.

### 2. Exploratory Data Analysis
- **analyzer_1.py**: Calculates descriptive statistics (e.g., average speed, counts) and generates basic plots (histograms, scatter plots).

### 3. Advanced Geospatial Analysis
- **analyzer_2.0.1.py**: Creates interactive maps and heatmaps based on geographic data.
- Uses **contextily**, **folium**, and **geopandas** to overlay data on real-world maps.

### 4. Clustering
- **clustering_1.0.1.py**, **clustering_1.0.2.py**, **clustering_1.0.3.py**: Applies clustering algorithms (K-Means, DBSCAN, HDBSCAN) to group similar trajectories.

### 5. Classification
- **classification_1.0.1.py**, **classification_1.0.2.py**, **classification_1.0.3.py**: Implements machine learning models (Random Forest, SVM, XGBoost) to classify vessel types or routes.

### 6. Prediction
- **prediction_9.0.1.py**: Uses regression models (Linear Regression, LSTM on time sequences) to predict future position and speed.

### 7. Bearing & Trajectory Features
- **analyzer_bearing.py**: Calculates bearing angles and trajectory-based features to enrich datasets.

### 8. Aggregated Statistics & Counts
- **counter_1.py**: Aggregates and analyzes parameters (e.g., vessel count by time period, vessel type distribution, temporal statistics).

