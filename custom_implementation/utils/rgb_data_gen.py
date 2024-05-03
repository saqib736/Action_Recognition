import csv
import os
import shutil

def organize_images_from_csv(csv_file, source_dir, target_dir):
    # Read the CSV file
    with open(csv_file, newline='') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row

        # Process each row in the CSV data
        for row in reader:
            file_name, label, sequence_id = row
            label = label.strip().replace(" ", "_").lower()  # Clean and format the label
            sequence_id = sequence_id.strip()

            # Define the directory path for the label and sequence
            label_dir = os.path.join(target_dir, label)
            sequence_dir = os.path.join(label_dir, sequence_id)

            # Ensure the label and sequence directories exist
            os.makedirs(sequence_dir, exist_ok=True)

            # Define the source and destination paths for the file
            src_path = os.path.join(source_dir, file_name)
            dest_path = os.path.join(sequence_dir, file_name)

            # Move the file from the source to the destination
            shutil.copy(src_path, dest_path)  # Use shutil.copy(src_path, dest_path) if you prefer to copy instead of move

# Define the paths
csv_file = 'filenames/label_data.csv'
source_dir = 'raw_data/thermal'
target_dir = 'Thermal_data'

# Call the function to organize images
organize_images_from_csv(csv_file, source_dir, target_dir)
