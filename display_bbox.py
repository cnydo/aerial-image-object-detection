import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import random
import yaml
import os
from tqdm import tqdm


def display_bbox(
    data_path,
    yaml_path,
    output_dir,
    only_with_bbox: bool = True,
    num_samples: int = 1,
    display: bool = False,
):
    # Load yaml file
    with open(yaml_path, "r") as f:
        yaml_dict = yaml.safe_load(f)
        class_names = yaml_dict["names"]
    # Get all image files
    img_dir = Path(data_path) / "images"
    img_files = list(img_dir.glob("*.jpg"))

    if only_with_bbox:
        # Filter to only include images with a corresponding bounding box file
        img_files = [
            img_file
            for img_file in img_files
            if (Path(data_path) / "labels" / img_file.with_suffix(".txt").name).exists()
        ]

    # Check if number of samples is greater than number of files
    if num_samples > len(img_files):
        raise ValueError(
            f"Number of samples ({num_samples}) is greater than number of available files ({len(img_files)})."
        )

    # Pick a number of random image files
    img_files = random.sample(img_files, num_samples)

    # Display images
    if only_with_bbox:
        for img_file in (pbar:=tqdm(img_files)):
            pbar.set_description(f"Drawing bounding boxes for {img_file.name}")
            # Read bounding box information from corresponding label file
            label_file = Path(data_path) / "labels" / img_file.with_suffix(".txt").name
            with open(label_file, "r") as f:
                bboxes = [
                    line.split() for line in f.readlines()
                ]  # Start reading from the second element

            # Read image
            img = cv2.imread(str(img_file.resolve()))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Draw bounding boxes on image
            for bbox in bboxes:
                class_index = int(bbox[0])
                class_name = class_names[class_index]
                bbox = bbox[1:]
                x_center, y_center, w, h = map(float, bbox)
                x_center, y_center, w, h = (
                    x_center * img.shape[1],
                    y_center * img.shape[0],
                    w * img.shape[1],
                    h * img.shape[0],
                )
                x = int(x_center - w / 2)
                y = int(y_center - h / 2)
                w = int(w)
                h = int(h)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    img,
                    class_name,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (36, 255, 12),
                    2,
                )
                # cv2.putText(img, f"Image name: {img_file.name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            # save image
            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(output_dir, img_file.name), img)
            # Display image
            if display:
                plt.imshow(img)
                plt.title(f"Image name: {img_file.name}")
                plt.axis("off")
                plt.show()
                
        print(f"Image saved to {output_dir}") if output_dir is not None else None
    else:  # Display images without bounding boxes
        for img_file in (pbar:=tqdm(img_files)):
            pbar.set_description(f"Drawing bounding boxes for {img_file.name}")
            img = cv2.imread(str(img_file.resolve()))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if display:
                plt.imshow(img)
                plt.title(f"Image name: {img_file.name}")
                plt.axis("off")
                plt.show()
            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(output_dir, img_file.name), img)
        print(f"Image saved to {output_dir}") if output_dir is not None else None


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    """
    - data_path: Path to the data directory. Must contain 'images' and 'labels' subdirectories.
    - yaml_path: Path to the yaml file with class names.
    - only_with_bbox: Only choose images that have bounding boxes.
    - num_samples: Number of sample images to display (default: 1)
    - output_dir: Path to the directory where the output files will be saved. 
    """
    parser = argparse.ArgumentParser(
        description="Display a sample image with bounding boxes."
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the data directory."
    )
    parser.add_argument(
        "--yaml_path",
        type=str,
        required=True,
        help="Path to the yaml file with class names.",
    )
    parser.add_argument("--output_dir", type=str, help="Path to the output directory.")
    parser.add_argument(
        "--only_with_bbox",
        default=True,
        type=str2bool,
        help="Only choose images that have bounding boxes.",
    )
    parser.add_argument(
        "--num_samples", type=int, default=1, help="Number of sample images to display."
    )
    # display_bbox
    parser.add_argument(
        "--display", default=False, type=str2bool, help="Display image."
    )
    args = parser.parse_args()
    display_bbox(
        args.data_path,
        args.yaml_path,
        args.output_dir,
        args.only_with_bbox,
        args.num_samples,
        args.display,
    )
