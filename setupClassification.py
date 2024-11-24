import os
import pandas as pd
import numpy as np
import multiprocessing as mp
from shapely.geometry import Point, Polygon
from tqdm import tqdm

# Define the polygon coordinates
polygon_coords = [
    [13.627332698482832, 46.01607627362711],
    [11.827331194451205, 45.56839707274841],
    [12.206288202161488, 44.06768319786181],
    [13.437866762923107, 43.47479960302897],
    [14.06942488652328, 42.20912261569302],
    [16.364148350192465, 43.688312063316346],
    [14.522085391698596, 44.50473436137818],
    [13.627332698482832, 46.01607627362711]  # Closing point of the polygon
]
area_polygon = Polygon(polygon_coords)

# Excluded MMSI prefixes and specific MMSIs
excluded_prefixes = [
    # Italy
    "992471",  # Physical AtoN
    "992476",  # Virtual AtoN
    "992478",  # Mobile AtoN
    # Croatia
    "992381",  # Physical AtoN
    "992386",  # Virtual AtoN
    "992388",  # Mobile AtoN
    # Slovenia
    "992781",  # Physical AtoN
    "992786",  # Virtual AtoN
    "992788",  # Mobile AtoN
]

mmsi_exclude = [
    "2470017",  # Italy
    "2470018",  # Italy
    "992467018",  # Italy
    "2470059",  # Italy
    "2470058",  # Italy
    "2470020",  # Italy
    "2780202",  # Slovenia
    "2386240",  # Croatia
    "2386300",  # Croatia
    "2386010",  # Croatia
    "2386020",  # Croatia
    "2386260",  # Croatia
    "2386190",  # Croatia
    "2386030",  # Croatia
    "2386080"   # Croatia
]

# Filter functions
def filter_by_polygon(chunk):
    if 'Longitude' not in chunk.columns or 'Latitude' not in chunk.columns:
        raise KeyError("Columns 'Longitude' and 'Latitude' are required.")
    return chunk[chunk.apply(lambda row: area_polygon.contains(Point(row['Longitude'], row['Latitude'])), axis=1)]

def filter_by_mmsi(chunk):
    if 'MMSI' not in chunk.columns:
        raise KeyError("Column 'MMSI' is required.")
    chunk['MMSI'] = chunk['MMSI'].astype(str)
    excluded_prefixes_set = set(excluded_prefixes)
    chunk = chunk[
        ~chunk['MMSI'].apply(lambda x: any(x.startswith(prefix) for prefix in excluded_prefixes_set)) &
        ~chunk['MMSI'].isin(mmsi_exclude)
    ]
    return chunk

def process_file(args):
    file_path, output_folder = args
    try:
        # Load the CSV file
        data = pd.read_csv(file_path)
        # Initialize list to collect filtered chunks
        filtered_data_list = []
        # Define chunk size for processing large files
        chunk_size = 100000  # Adjust as needed

        # Process data in chunks
        for chunk_start in range(0, data.shape[0], chunk_size):
            chunk_end = min(chunk_start + chunk_size, data.shape[0])
            chunk = data.iloc[chunk_start:chunk_end]
            # Apply filters
            chunk = filter_by_polygon(chunk)
            chunk = filter_by_mmsi(chunk)
            filtered_data_list.append(chunk)

        # Concatenate all filtered chunks
        if filtered_data_list:
            filtered_data = pd.concat(filtered_data_list, ignore_index=True)
            # Save the filtered data to the output folder
            output_file_path = os.path.join(output_folder, os.path.basename(file_path))
            filtered_data.to_csv(output_file_path, index=False)
        else:
            print(f"No data left after filtering for file {file_path}")

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")


if __name__ == '__main__':
    # Path to the dataset folder containing the original CSV files
    dataset_folder = "dataset/AIS_Dataset_csv"

    # List of CSV files to process
    csv_files = [os.path.join(dataset_folder, f) for f in os.listdir(dataset_folder) if f.endswith('.csv')]

    # Create the output folder
    output_folder = os.path.join("dataset", "AIS_Dataset_csv_FocusArea")
    os.makedirs(output_folder, exist_ok=True)

    # Number of processes to use
    num_processes = mp.cpu_count()

    # Prepare arguments for the process_file function
    file_args = [(file_path, output_folder) for file_path in csv_files]

    # Process files in parallel
    with mp.Pool(processes=num_processes) as pool:
        list(tqdm(pool.imap_unordered(process_file, file_args), total=len(csv_files), desc="Processing files"))

    print("All files processed and saved in 'AIS_Dataset_csv_FocusArea'.")
