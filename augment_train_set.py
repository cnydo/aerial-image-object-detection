import cv2
from albumentations import Compose, RandomBrightnessContrast, RandomRain
import matplotlib.pyplot as plt
from IPython.display import Image, display
import random
from shutil import copy
from pathlib import Path
from tqdm import tqdm


def augment_brightness(image, darkenings=False):
    brightness_limit = (-0.2, -0.2) if darkenings else (0.15, 0.15)
    augment = Compose(
        [RandomBrightnessContrast(brightness_limit=brightness_limit, p=1)]
    )
    augmented = augment(image=image)
    image = augmented["image"]
    return image


def augment_rain(image):
    augment = Compose(
        [
            RandomRain(
                brightness_coefficient=0.9,
                drop_width=1,
                blur_value=5,
                p=1,
                rain_type="drizzle",
            )
        ]
    )
    augmented = augment(image=image)
    image = augmented["image"]
    return image


def augment_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return (
        augment_brightness(image, darkenings=True),
        augment_brightness(image, darkenings=False),
        augment_rain(image),
    )


def augment_images(input_folder, output_folder):
    output_folder_path = Path(output_folder)
    output_folder_path.mkdir(parents=True, exist_ok=True)
    for label_file in (pbar := tqdm(Path(input_folder).glob("labels/*.txt"))):
        pbar.set_description(f"Processing {label_file.stem}")
        image_file = Path(input_folder) / "images" / (label_file.stem + ".jpg")
        if image_file.exists():
            # create output folders
            (output_folder_path / image_file.parent.name).mkdir(
                parents=True, exist_ok=True
            )
            (output_folder_path / label_file.parent.name).mkdir(
                parents=True, exist_ok=True
            )

            # save augment images
            for augment_type in ["darkened", "brightened", "rain"]:
                image_augment_path = (
                    output_folder_path
                    / "images"
                    / f"{image_file.stem}_{augment_type}{image_file.suffix}"
                )
                label_augment_path = (
                    output_folder_path
                    / "labels"
                    / f"{label_file.stem}_{augment_type}{label_file.suffix}"
                )
                img = cv2.imread(str(image_file.resolve()))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if augment_type == "darkened":
                    img = augment_brightness(img, darkenings=True)
                elif augment_type == "brightened":
                    img = augment_brightness(img, darkenings=False)
                elif augment_type == "rain":
                    img = augment_rain(img)
                # convert back to BGR
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(image_augment_path.resolve()), img)
                # copy labels of the original image to the augmented images
                copy(label_file, label_augment_path)
    print("Augmented images saved in", output_folder_path)

if __name__ == "__main__":
    augment_images("train", "train")
