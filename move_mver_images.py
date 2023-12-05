import shutil
from pathlib import Path
from tqdm.auto import tqdm
def move_images_from_sub_to_parent(sub_directory):
    sub_directory_path = Path(sub_directory)
    parent_directory_path = sub_directory_path.parent
    list_of_image_files = list(sub_directory_path.glob("*.jpg"))
    for image_file in (pbar:=tqdm(list_of_image_files)):
        pbar.set_description(f"Moving {image_file.name:30s}")
        shutil.move(str(image_file), str(parent_directory_path))
    # remove the sub directory
    shutil.rmtree(str(sub_directory_path))
    
if __name__ == "__main__":
    move_images_from_sub_to_parent("MVER")