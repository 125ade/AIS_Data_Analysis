# AIS Data Analysis

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

Run the following script to organize and prepare the vessels tracks CSV files:

```bash
python setupTracks.py
```

### Step 6: Run Data Analysis

Initiate a preliminary analysis of the dataset using:

```bash
python analyzer.py
```
