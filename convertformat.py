import argparse
import os
import shutil
import sys
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

# Define the paths to the output directories
output_dirs = ["train/labels", "test/labels", "val/labels"]
mapping_dirs = {
    "annotations_train.csv": "train/labels",
    "annotations_test.csv": "test/labels",
    "annotations_val.csv": "val/labels",
}

def process_file(input_path, class_names: list = ["Zebra", "Giraffe", "Elephant"]):
    # Create the output directories if they don't exist
    input_path = Path(input_path)
    output_dir = Path(mapping_dirs[input_path.name])
    if output_dir.exists():
        print(f"{output_dir} already exists. Deleting...")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read the input file using pandas
    df = pd.read_csv(input_path, header=None)

    # Create a tqdm progress bar for the loop
    progress_bar = tqdm(df.iterrows(), total=len(df))
    progress_bar.set_description(f"Processing {str(input_path):20s}")
    # Loop over each row in the CSV file
    for i, row in progress_bar:
        # Extract the image filename and the bounding box coordinates
        image_filename, *box, class_name = row
        x_min, y_min, x_max, y_max = map(float, box)

        # Calculate the center coordinates and the width and height of the bounding box
        image = Image.open(image_filename)
        image_width, image_height = image.size
        x_center = (x_min + x_max) / 2 / image_width
        y_center = (y_min + y_max) / 2 / image_height
        width = (x_max - x_min) / image_width
        height = (y_max - y_min) / image_height

        # Look up the class index based on the class name
        class_index = class_names.index(class_name)

        # Create the output file path        
        image_path = Path(image_filename)
        output_path = image_path.parent / "labels" / f"{image_path.stem}.txt"

        # Write the annotation in YOLOv8 format to the output file
        with open(output_path, "a") as output_file:
            output_file.write(f"{class_index} {x_center} {y_center} {width} {height}\n")

        # Update the progress bar
        progress_bar.set_description(f"Processing {image_filename}")

    # Print results and output file paths
    print(f"Processed {len(df)} rows. Output files saved to {output_dir.resolve()}")


if __name__ == "__main__":
    # Get the input file paths from the command line arguments
    parser = argparse.ArgumentParser(
        description="Convert CSV annotations to YOLOv8 format."
    )
    parser.add_argument(
        "input_files",
        metavar="input_file",
        type=str,
        nargs="+",
        help="an input CSV file",
    )
    parser.add_argument(
        "--class_names",
        metavar="class_names",
        type=str,
        nargs="+",
        default=["Zebra", "Giraffe", "Elephant"],
        help="a list of class names",
    )
    args = parser.parse_args()
    # class_names = args.class_names.split(',')
    # Process each input file
    for input_file in args.input_files:
        process_file(input_file, args.class_names)
