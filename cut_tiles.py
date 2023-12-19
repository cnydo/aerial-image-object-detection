import argparse
from math import ceil
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
from PIL import Image
from tqdm import tqdm

TILES_DIR = {
    "train": Path("train/images/"),
    "val": Path("val/images/"),
    "test": Path("test/images/"),
}
for _, folder in TILES_DIR.items():
    if not folder.is_dir():
        folder.mkdir(parents=True, exist_ok=True)

LABELS_DIR = {
    "train": Path("train/labels/"),
    "val": Path("val/labels/"),
    "test": Path("test/labels/"),
}
for _, folder in LABELS_DIR.items():
    if not folder.is_dir():
        folder.mkdir(parents=True, exist_ok=True)


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


# Read images and labels from the folder
def read_images_and_labels(folder_path) -> pd.DataFrame:
    folder_path = Path(folder_path)
    data = []

    for image_file in (pbar := tqdm(folder_path.glob("images/*.jpg"))):
        pbar.set_description(f"Reading images and labels for {image_file.name:30s}")
        image_name = image_file.stem
        label_file = folder_path / "labels" / (image_name + ".txt")
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
def cut_tile(
    data_dir: str,
    tile_width: int = 900,
    tile_height: int = 900,
    tile_overlap: int = 200,
    truncated_percent: float = 0.1,
    _overwriteFiles: bool = True,
):
    """The `cut_tile` function is used to create tiles and save them with their labels into the corresponding folders (\images and \labels).

    Args:
        data_dir (str): _description_
        tile_width (int, optional): _description_. Defaults to 900.
        tile_height (int, optional): _description_. Defaults to 900.
        tile_overlap (int, optional): _description_. Defaults to 200.
        truncated_percent (float, optional): _description_. Defaults to 0.1.
        _overwriteFiles (bool, optional): _description_. Defaults to True.
    """
    data_dir = Path(data_dir)
    image_list = list(data_dir.glob("images/*.jpg"))

    df = read_images_and_labels(data_dir)
    for img_path in (pbar := tqdm(image_list)):
        pbar.set_description(f"Cutting tiles for {img_path.name:30s}")
        # Open image and related data
        pil_img = Image.open(img_path, mode="r")
        img_width, img_height = pil_img.size
        np_img = np.array(pil_img, dtype=np.uint8)
        # Get annotations for image
        img_labels = df[df["image_name"] == str(img_path.stem)]
        # find the number of tiles that can fit in the image with overlapping
        X_TILES = get_num_tiles(img_width, tile_width, tile_overlap)
        Y_TILES = get_num_tiles(img_height, tile_height, tile_overlap)

        # Cut each tile
        for x in range(X_TILES):
            for y in range(Y_TILES):
                x_end = min((x + 1) * tile_width - tile_overlap * (x), img_width)
                x_start = x_end - tile_width
                y_end = min((y + 1) * tile_height - tile_overlap * (y), img_height)
                y_start = y_end - tile_height
                folder = img_path.parent.parent.name
                save_tile_path = TILES_DIR[folder].joinpath(
                    img_path.stem + "_" + str(x_start) + "_" + str(y_start) + ".jpg"
                )
                save_label_path = LABELS_DIR[folder].joinpath(
                    img_path.stem + "_" + str(x_start) + "_" + str(y_start) + ".txt"
                )

                # Save if file doesn't exit
                if _overwriteFiles or not Path(save_label_path).is_file():
                    cut_tile = np.zeros(
                        shape=(tile_width, tile_height, 3), dtype=np.uint8
                    )
                    cut_tile[0:tile_height, 0:tile_width, :] = np_img[
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
                        tile_width,
                        tile_height,
                        truncated_percent,
                    )
                    found_tags.append(tag)
                found_tags = [el for el in found_tags if el is not None]
                # save labels
                if len(found_tags) > 0:
                    with open(save_label_path, "w+") as f:
                        for tags in found_tags:
                            f.write(" ".join(str(x) for x in tags) + "\n")

    remove_raw_images_and_labels(image_list)


def remove_raw_images_and_labels(img_list: list):
    for img_path in (pbar := tqdm(img_list)):
        pbar.set_description(f"Removing raw images and labels for {img_path.name:30s}")
        # Remove image
        Path(img_path).unlink(missing_ok=True)

        # Remove label
        label_path = (
            Path(f"{img_path.parent.parent.name}/labels/") / img_path.stem
        ).with_suffix(".txt")
        Path(label_path).unlink(missing_ok=True)
    print("Done!")


def draw_tiles_on_image(img_path):
    """The `draw_tiles_on_image` function is used to draw tiles on the image.

    Args:
        img_path (_type_): _description_
    """
    # Read the image
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Get the dimensions of the image
    IMAGE_HEIGHT, IMAGE_WIDTH, _ = img.shape
    TILE_HEIGHT = TILE_WIDTH = 900
    TILE_OVERLAP = 200
    # Calculate the number of tiles
    X_TILES = get_num_tiles(IMAGE_WIDTH, TILE_WIDTH, TILE_OVERLAP)
    Y_TILES = get_num_tiles(IMAGE_HEIGHT, TILE_HEIGHT, TILE_OVERLAP)
    # Draw rectangles (tiles) on the image
    # RGB color for the border
    border_color = (0, 255, 0)
    border_thickness = 15  # Thickness of the border
    print(X_TILES, Y_TILES)
    for x in range(X_TILES):
        for y in range(Y_TILES):
            x_end = min((x + 1) * TILE_WIDTH - TILE_OVERLAP * (x), IMAGE_WIDTH)
            x_start = x_end - TILE_WIDTH
            y_end = min((y + 1) * TILE_HEIGHT - TILE_OVERLAP * (y), IMAGE_HEIGHT)
            y_start = y_end - TILE_HEIGHT

            cv2.rectangle(
                img, (x_start, y_start), (x_end, y_end), border_color, border_thickness
            )
            print(f"{x_start = }, {y_start = }, {x_end = }, {y_end = }")
    # Save the image with the border
    cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("image_with_tiles_DPP_00418.jpg", img)
    plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cut tiles from images")
    parser.add_argument(
        "folder_path",
        type=str,
        help="Path to the folder containing images and labels",
    )
    parser.add_argument(
        "--tile_width",
        type=int,
        default=900,
        help="Width of the tile",
    )
    parser.add_argument(
        "--tile_height",
        type=int,
        default=900,
        help="Height of the tile",
    )
    parser.add_argument(
        "--tile_overlap",
        type=int,
        default=200,
        help="Overlap between tiles",
    )
    parser.add_argument(
        "--truncated_percent",
        type=float,
        default=0.1,
        help="Truncated percentage",
    )
    parser.add_argument(
        "--overwriteFiles",
        type=bool,
        default=True,
        help="Overwrite existing files",
    )
    args = parser.parse_args()
    cut_tile(
        args.folder_path,
        args.tile_width,
        args.tile_height,
        args.tile_overlap,
        args.truncated_percent,
        args.overwriteFiles,
    )
