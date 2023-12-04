import shutil
from pathlib import Path
from tqdm.auto import tqdm
def move_images_from_sub_to_parent(sub_directory):
    sub_directory_path = Path(sub_directory)
    parent_directory_path = sub_directory_path.parent

    for image_file in tqdm(sub_directory_path.glob("*"), desc=f"Moving images from {sub_directory} to {parent_directory_path}"):
        shutil.move(str(image_file), str(parent_directory_path))
    # remove the sub directory
    shutil.rmtree(str(sub_directory_path))
move_images_from_sub_to_parent("MVER")