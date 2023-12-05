from pathlib import Path
from tqdm.auto import tqdm
from math import ceil
from PIL import Image
import random
import pandas as pd
import os
from tqdm.auto import tqdm
from math import ceil
import numpy as np
import pandas as pd

DATA_DIR = Path("val")
val_indexes = list(DATA_DIR.glob("images/*.jpg"))


# Function to convert labels from yolo xywh format to xyxy format
def yolo_to_corners(yolo_format: list | tuple | set) -> tuple:
    """The `yolo_to_corners` function is used to convert labels from yolo format to corner format.
    (x_center, y_center, width, height) -> (top_left_x, top_left_y, bottom_right_x, bottom_right_y

    Args:
        yolo_format (_type_): yolo format labels (x_center, y_center, width, height)

    Returns:
        _type_: bounding box coordinates (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
    """
    x_center, y_center, width, height = yolo_format
    top_left_x = x_center - (width / 2)
    top_left_y = y_center - (height / 2)
    bottom_right_x = x_center + (width / 2)
    bottom_right_y = y_center + (height / 2)
    return (top_left_x, top_left_y, bottom_right_x, bottom_right_y)


def read_images_and_labels(val_folder):
    val_folder = Path(val_folder)
    data = []

    for image_file in tqdm(val_folder.glob("images/*.jpg")):
        image_name = image_file.stem
        label_file = val_folder / "labels" / (image_name + ".txt")
        img_width, img_height = Image.open(image_file).size

        if label_file.exists():
            with open(label_file, "r") as file:
                for line in file:
                    class_id, x_center, y_center, width, height = map(
                        float, line.split()
                    )
                    x_center *= img_width
                    y_center *= img_height
                    width *= img_width
                    height *= img_height
                    (
                        top_left_x,
                        top_left_y,
                        bottom_right_x,
                        bottom_right_y,
                    ) = yolo_to_corners((x_center, y_center, width, height))
                    data.append(
                        [
                            image_name,
                            class_id,
                            top_left_x,
                            top_left_y,
                            bottom_right_x,
                            bottom_right_y,
                        ]
                    )

    df = pd.DataFrame(
        data,
        columns=[
            "image_name",
            "class_id",
            "top_left_x",
            "top_left_y",
            "bottom_right_x",
            "bottom_right_y",
        ],
    )
    return df


TILE_WIDTH = 900
TILE_HEIGHT = 900
TILE_OVERLAP = 200
TRUNCATED_PERCENT = 0.1  # minimum threshold to consider a bounding box as truncated
_overwriteFiles = True

TILES_DIR = {"train": Path("train/images/"), "val": Path("val/images/")}
for _, folder in TILES_DIR.items():
    if not os.path.isdir(folder):
        os.makedirs(folder)

LABELS_DIR = {"train": Path("train/labels/"), "val": Path("val/labels/")}
for _, folder in LABELS_DIR.items():
    if not os.path.isdir(folder):
        os.makedirs(folder)


# Function to check if the tag is inside the tile
def tag_is_inside_tile(
    bounds: int,
    x_start: int,
    y_start: int,
    tile_width: int,
    tile_height: int,
    truncated_percent: float,
) -> tuple | None:
    """The `tag_is_inside_tile` function is used to determine if a tag (bounding box) is completely or partially inside a tile.

    Args:
        bounds (_type_): bounding box coordinates (x_min, y_min, x_max, y_max)
        x_start (_type_): starting x coordinate of the tile
        y_start (_type_): starting y coordinate of the tile
        width (_type_): width of the tile
        height (_type_): height of the tile
        truncated_percent (_type_): truncated percentage

    Returns:
        _type_: _description_
    """
    class_id, x_min, y_min, x_max, y_max = bounds
    x_min, y_min, x_max, y_max = (
        x_min - x_start,
        y_min - y_start,
        x_max - x_start,
        y_max - y_start,
    )

    # The condition checks if the bounding box is completely outside the tile.
    if (x_min > tile_width) or (x_max < 0.0) or (y_min > tile_height) or (y_max < 0.0):
        return None

    # Determine if a tag (bounding box) is completely or partially inside a tile.
    x_max_trunc = min(x_max, tile_width)
    x_min_trunc = max(x_min, 0)
    if (x_max_trunc - x_min_trunc) / (x_max - x_min) < truncated_percent:
        return None

    y_max_trunc = min(y_max, tile_width)
    y_min_trunc = max(y_min, 0)
    if (y_max_trunc - y_min_trunc) / (y_max - y_min) < truncated_percent:
        return None

    # Calculate the relative coordinates of the bounding box in the tile.
    x_center = (x_min_trunc + x_max_trunc) / 2.0 / tile_width
    y_center = (y_min_trunc + y_max_trunc) / 2.0 / tile_height
    bbox_width = (x_max_trunc - x_min_trunc) / tile_width
    bbox_height = (y_max_trunc - y_min_trunc) / tile_height

    return (int(class_id), x_center, y_center, bbox_width, bbox_height)


# Function to calculate the number of tiles that can fit in the image with overlapping
def get_num_tiles(image_size: int, tile_size: int, tile_overlap: int) -> int:
    """The `get_num_tiles` function is used to calculate the number of tiles that can fit in the image with overlapping.

    Args:
        image_size (int): image size (width or height)
        tile_size (int): tile size (width or height)
        tile_overlap (int): tile overlap (width or height)

    Returns:
        int: number of tiles that can fit in the image with overlapping
    """
    num_tiles_not_overlapping = ceil(image_size / tile_size)
    remainder = image_size - (
        num_tiles_not_overlapping * tile_size
        - tile_overlap * (num_tiles_not_overlapping - 1)
    )
    return ceil(image_size / tile_size) + ceil((remainder) / (tile_size - tile_overlap))


# Create tiles and save them ưith their labels into the corresponding folders (val\images and val\labels)
def cut_tile():
    df = read_images_and_labels(DATA_DIR)
    for img_path in tqdm.notebook.tqdm(val_indexes):
        # Open image and related data
        pil_img = Image.open(img_path, mode="r")
        IMAGE_WIDTH, IMAGE_HEIGHT = pil_img.size
        np_img = np.array(pil_img, dtype=np.uint8)
        # Get annotations for image
        img_labels = df[df["image_name"] == str(img_path.stem)]
        # find the number of tiles that can fit in the image with overlapping
        X_TILES = get_num_tiles(IMAGE_WIDTH, TILE_WIDTH, TILE_OVERLAP)
        Y_TILES = get_num_tiles(IMAGE_HEIGHT, TILE_HEIGHT, TILE_OVERLAP)

        # Cut each tile
        for x in range(X_TILES):
            for y in range(Y_TILES):
                x_end = min((x + 1) * TILE_WIDTH - TILE_OVERLAP * (x), IMAGE_WIDTH)
                x_start = x_end - TILE_WIDTH
                y_end = min((y + 1) * TILE_HEIGHT - TILE_OVERLAP * (y), IMAGE_HEIGHT)
                y_start = y_end - TILE_HEIGHT
                # print(x_start, y_start)

                # folder = "val" if img_path.name in val_indexes else "train"
                folder = "val"
                save_tile_path = TILES_DIR[folder].joinpath(
                    img_path.stem + "_" + str(x_start) + "_" + str(y_start) + ".jpg"
                )
                save_label_path = LABELS_DIR[folder].joinpath(
                    img_path.stem + "_" + str(x_start) + "_" + str(y_start) + ".txt"
                )

                # Save if file doesn't exit
                if _overwriteFiles or not os.path.isfile(save_tile_path):
                    cut_tile = np.zeros(
                        shape=(TILE_WIDTH, TILE_HEIGHT, 3), dtype=np.uint8
                    )
                    cut_tile[0:TILE_HEIGHT, 0:TILE_WIDTH, :] = np_img[
                        y_start:y_end, x_start:x_end, :
                    ]
                    cut_tile_img = PIL.Image.fromarray(cut_tile)
                    cut_tile_img.save(save_tile_path)

                # found_tags = [
                #     tag_is_inside_tile(
                #         bounds, x_start, y_start, TILE_WIDTH, TILE_HEIGHT, TRUNCATED_PERCENT
                #     )
                #     for () in img_labels.iterrows[['top_left_x', 'top_left_y', 'bottom_right_x', 'bottom_right_y']]
                # ]
                found_tags = []
                for row in img_labels[
                    [
                        "class_id",
                        "top_left_x",
                        "top_left_y",
                        "bottom_right_x",
                        "bottom_right_y",
                    ]
                ].iterrows():
                    bounds = tuple([row[1][i] for i in range(5)])
                    tag = tag_is_inside_tile(
                        bounds,
                        x_start,
                        y_start,
                        TILE_WIDTH,
                        TILE_HEIGHT,
                        TRUNCATED_PERCENT,
                    )
                    found_tags.append(tag)
                found_tags = [el for el in found_tags if el is not None]
                # save labels
                if len(found_tags) > 0:
                    with open(save_label_path, "w+") as f:
                        for tags in found_tags:
                            f.write(" ".join(str(x) for x in tags) + "\n")


def remove_val_raw_image():
    for img_path in tqdm(val_indexes):
        # Remove image
        Path(img_path).unlink(missing_ok=True)

        # Remove label
        label_path = (Path("val/labels/") / img_path.stem).with_suffix(
            ".txt"
        )  # replace with your label file extension
        Path(label_path).unlink(missing_ok=True)


if __name__ == "__main__":
    cut_tile()
    remove_val_raw_image()
