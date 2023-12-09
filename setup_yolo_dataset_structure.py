from tqdm import tqdm
import shutil
from pathlib import Path
import comet_ml

# Directories to process
directories = ['train', 'val', 'test']

for directory in directories:
    # Create 'images' subdirectory
    Path(directory, 'images').mkdir(exist_ok=True)

    # Get list of .jpg files in directory
    images = list(Path(directory).glob('*.jpg'))
    if len(images) == 0:
        print(f"No .jpg files found in {directory}")
        continue
    # Move .jpg files to 'images' subdirectory with progress bar
    for image_path in (pbar:=tqdm(images)):
        pbar.set_description(f"Moving {image_path.name:30s} into {directory}/images")
        shutil.move(str(image_path), str(Path(directory, 'images', image_path.name)))