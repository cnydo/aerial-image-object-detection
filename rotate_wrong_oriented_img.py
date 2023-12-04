import cv2
from pathlib import Path
from tqdm import tqdm

# Read image paths from a text file
with open('list of wrong oriented image.txt', 'r') as f:
    image_paths = [line.strip() for line in f]

for image_path_str in (progress_bar:=tqdm(image_paths)):
    progress_bar.set_description(f"Processing {image_path_str}")
    image_path = Path(image_path_str)
    if image_path.exists():
        img = cv2.imread(str(image_path))
        rotated_img = cv2.rotate(img, cv2.ROTATE_180)
        cv2.imwrite(str(image_path.resolve()), rotated_img)
        